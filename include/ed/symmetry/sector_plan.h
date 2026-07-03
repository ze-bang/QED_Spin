#pragma once
// =============================================================================
// include/ed/symmetry/sector_plan.h
//
// Stage 8 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): the COMPOSITION layer.
//
// Stages 5-6 introduced three symmetry-exploitation mechanisms:
//
//   * spin-flip TRANSPORT   solve n_up <= N/2, mirror results to N - n_up
//   * spin-flip PROJECTION  split the n_up = N/2 block into (k, +/-)
//   * time-reversal PAIRING solve one of {k, -k}, copy to the partner
//
// but the orchestration lived hand-wired inside ONE consumer (the
// all-Sz flat-pool thermal binding). This header lifts it into pure,
// unit-testable planning functions so EVERY workflow (thermal
// recombination, ground-state eigenvalue pooling, spectral sector
// loops) composes the same way:
//
//   1. resolve_symmetry_composition(H terms, group, backend, toggles)
//        -> which mechanisms apply (auto-detected commutation checks,
//           per-call override modes, env gates as the Auto defaults).
//   2. plan_build_window(comp, requested n_up window, N)
//        -> which Sz blocks to BUILD (transport clamps to <= N/2) and
//           whether the half-filling block is flip-projected.
//   3. plan_tr_actions(tags, num_raw_sectors, comp)
//        -> per built sector: SOLVE or COPY-from-partner (index map).
//   4. flip_mirror_target(n_up, N) + window predicates
//        -> post-collection mirror emission for the transported blocks.
//
// Everything here is a pure function of its inputs -- no I/O, no env
// reads except through the explicit Auto-mode defaults -- pinned by
// tests/unit/test_sector_plan.cpp.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ed/core/symmetry_metadata.h>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/time_reversal.h>

namespace ed::symmetry {

/// Per-call override for one symmetry mechanism. ``Auto`` defers to the
/// commutation check + the env gate (ED_SYM_SPIN_FLIP / _PROJECT /
/// ED_SYM_TIME_REVERSAL); ``Off`` disables unconditionally; ``Require``
/// throws with a diagnostic when the Hamiltonian does not actually
/// carry the symmetry (loud is better than silently-different physics).
enum class SymToggle : std::int8_t { Off = 0, Require = 1, Auto = -1 };

[[nodiscard]] inline SymToggle sym_toggle_from_int(int v) noexcept {
    return v == 0 ? SymToggle::Off
         : v == 1 ? SymToggle::Require
                  : SymToggle::Auto;
}

struct SymmetryComposition {
    // Detected properties of H (informational; independent of toggles).
    bool flip_symmetric = false;   // [H, prod sigma^x] == 0 at term level
    bool h_real         = false;   // every coefficient real

    // Active mechanisms after toggles + env + backend gating.
    bool flip_transport = false;
    bool flip_project   = false;   // n_up = N/2 in-sector (k, +/-) split
    bool tr_pairing     = false;

    // Conjugate-sector map (raw irrep -> raw irrep); empty unless
    // tr_pairing. partner[k] == -1 marks malformed metadata for k
    // (that sector is always solved).
    std::vector<std::int32_t> tr_partner;
};

/// Resolve which mechanisms apply. All three mechanisms are
/// backend-independent as of Stage 8b (the device rep policy carries
/// the same per-element flip masks as the host policy); ``allow_gpu``
/// is retained in the signature for future backend-specific gating.
[[nodiscard]] inline SymmetryComposition
resolve_symmetry_composition(const ed::matvec::TermStorage& soa,
                             const SymmetryGroupInfo&       info,
                             bool                           allow_gpu,
                             SymToggle flip_mode = SymToggle::Auto,
                             SymToggle tr_mode   = SymToggle::Auto)
{
    SymmetryComposition c;
    c.flip_symmetric = hamiltonian_is_spin_flip_symmetric(soa);
    c.h_real         = hamiltonian_is_real(soa);

    // ---- spin flip ----
    switch (flip_mode) {
        case SymToggle::Off:
            break;
        case SymToggle::Require:
            if (!c.flip_symmetric) {
                throw std::runtime_error(
                    "symmetry composition: spin_flip='require' but "
                    "[H, prod sigma^x] != 0 at the term level (Zeeman "
                    "field, unpaired S+/-, or three-body content).");
            }
            c.flip_transport = true;
            break;
        case SymToggle::Auto:
            c.flip_transport =
                c.flip_symmetric && spin_flip_transport_enabled();
            break;
    }
    (void)allow_gpu;  // Stage 8b: projection is backend-independent now
    if (c.flip_transport) {
        const char* v = std::getenv("ED_SYM_SPIN_FLIP_PROJECT");
        c.flip_project = !(v != nullptr && v[0] == '0' && v[1] == '\0');
        // Projection exists only on the CSR-free rep lane: a
        // flip-projected sector has no orbit-CSR form (that provider
        // throws by design), so when the user pinned the legacy orbit
        // lane (ED_SYM_REP=0) the plan must not select it.
        if (!cpu_rep_symmetry_enabled()) c.flip_project = false;
    }

    // ---- time reversal ----
    switch (tr_mode) {
        case SymToggle::Off:
            break;
        case SymToggle::Require:
            if (!c.h_real) {
                throw std::runtime_error(
                    "symmetry composition: time_reversal='require' but H "
                    "has complex matrix elements in the computational "
                    "basis (imaginary coefficients).");
            }
            c.tr_pairing = true;
            break;
        case SymToggle::Auto:
            c.tr_pairing = c.h_real && time_reversal_pairing_enabled();
            break;
    }
    if (c.tr_pairing && !info.sectors.empty()) {
        c.tr_partner = conjugate_sector_pairing(info);
    } else {
        c.tr_pairing = false;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Build window: which Sz blocks to construct.
// ---------------------------------------------------------------------------
struct BuildWindow {
    int  lo = 0;
    int  hi = 0;
    bool flip_project_half = false;  // pass-through to the all-Sz builder
    bool flip_transport    = false;  // mirror emission required after solve
};

[[nodiscard]] inline BuildWindow
plan_build_window(const SymmetryComposition& comp,
                  int req_lo, int req_hi, int n_sites)
{
    BuildWindow w;
    w.lo = req_lo < 0 ? 0 : req_lo;
    w.hi = (req_hi < 0 || req_hi > n_sites) ? n_sites : req_hi;
    w.flip_transport    = comp.flip_transport;
    w.flip_project_half = comp.flip_project;
    if (comp.flip_transport) {
        const int lo2 = std::min(w.lo, n_sites - w.hi);
        const int hi2 = std::min(w.hi, n_sites / 2);
        if (hi2 < (lo2 < 0 ? 0 : lo2)) {
            // Degenerate request window: transport cannot cover it.
            w.flip_transport    = false;
            w.flip_project_half = false;
        } else {
            w.lo = lo2 < 0 ? 0 : lo2;
            w.hi = hi2;
        }
    }
    return w;
}

/// Flip partner of an Sz block.
[[nodiscard]] inline int flip_mirror_target(int n_up, int n_sites) noexcept {
    return n_sites - n_up;
}

// ---------------------------------------------------------------------------
// Time-reversal action plan over a BUILT sector list.
//
// ``Tags`` is any random-access range whose elements expose ``n_up``,
// ``sector_index`` (raw irrep index; flip-projected sectors use the
// synthetic k + parity * num_raw convention) and ``sector_dim``
// (ed::SectorTag satisfies this).
// ---------------------------------------------------------------------------
struct TrActionPlan {
    std::vector<char>        skip;    // per built sector: copy instead of solve
    std::vector<std::size_t> source;  // valid where skip[i] != 0
    std::size_t              n_skipped = 0;

    [[nodiscard]] bool active() const noexcept { return n_skipped > 0; }
};

template <class Tags>
[[nodiscard]] TrActionPlan
plan_tr_actions(const Tags& tags, std::size_t num_raw,
                const SymmetryComposition& comp)
{
    TrActionPlan plan;
    plan.skip.assign(tags.size(), 0);
    plan.source.assign(tags.size(), 0);
    if (!comp.tr_pairing || num_raw == 0 || comp.tr_partner.empty()) {
        return plan;
    }

    std::map<std::pair<int, std::size_t>, std::size_t> where;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        where[{tags[i].n_up, tags[i].sector_index}] = i;
    }
    for (std::size_t i = 0; i < tags.size(); ++i) {
        const auto& tag = tags[i];
        const std::size_t k = tag.sector_index % num_raw;
        const std::int32_t pk = comp.tr_partner[k];
        if (pk < 0) continue;                       // malformed: solve
        const std::size_t praw =
            static_cast<std::size_t>(pk) + (tag.sector_index / num_raw) * num_raw;
        if (praw >= tag.sector_index) continue;     // canonical member
        const auto it = where.find({tag.n_up, praw});
        if (it == where.end()) continue;            // partner absent: solve
        if (tags[it->second].sector_dim != tag.sector_dim) continue;  // paranoia
        plan.skip[i]   = 1;
        plan.source[i] = it->second;
        ++plan.n_skipped;
    }
    return plan;
}

}  // namespace ed::symmetry
