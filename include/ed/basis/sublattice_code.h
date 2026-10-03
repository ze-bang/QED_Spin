#pragma once
// =============================================================================
// include/ed/basis/sublattice_code.h -- the representative search by sublattices (P7.6).
//
// The representative of an orbit is its smallest member. Found by scanning the group, it costs
// |G| images of ceil(N/8) table lookups each (2160 lookups at the 36-site triangular space
// group). When the group permutes the blocks of a block system -- sublattices: every element maps
// each block onto a block, as the translations and point-group operations of a lattice permute
// the sublattices of a superlattice -- number the sites in a key order where block j holds key
// bits [N - (j + 1) L, N - j L). The leading L key bits of an image then come from one block of the
// state, through a table. The representative is the member smallest in the key order: its leading
// block is the least of m table entries, and only the elements that reach that leading block --
// usually one or a few -- need a full image (Wietek and Lauchli, Phys. Rev. E 98, 033309 (2018)).
//
// The key order decides which member of an orbit represents it -- not the orbits, their norms or
// anything physical -- but every canonicalisation of one sector must agree with the table its
// representatives came from. The code is a function of the group's distinct site permutations and
// of ED_SYM_SUBLATTICE alone (a group with or without its spin-flip half gets the same key order),
// read when a CompiledGroup is built; the orbit table records the code it used, and every sector
// built from the table carries that code (RepSectorData::slc) -- its policies and device mirrors
// read it there, never the environment, and a saved sector stores its fingerprint.
//
// ED_SYM_SUBLATTICE: unset -> on for N >= 24 sites with at least 16 distinct permutations under
// the relaxed rule (a verb on the device; eigs, spectrum and exact thermodynamics on the host) and
// 64 under the strict one (host sampled thermodynamics and host dynamics); 1 -> whenever a block
// system exists; 0 -> never. The key-least representatives cost a host sparse apply cache locality
// where the plain order had it -- a chain, whose bonds join neighbouring bits: chain30 k-sector
// +50% an apply, chain28 FTLM +18% wall (jobs 62698687, 62703856) -- and little or nothing on 2D
// lattices (tri36 +11% an apply, tri30 none) or on the device (chain32 the same; 62697934). The
// canonicalisation it saves (the CSR build 5x, the orbit table 2.5x, the device gather 1.8x) pays
// on the device always, on the host whenever H is applied a few times per block (tri30 eigs 74.8
// -> 54.9 s; even chain30 eigs 1.33 -> 1.21 s), and for thousands of host applies only on the large
// groups of 2D space groups. The physics is the same either way; the representative basis differs
// by phases, so seeded sampled results move at sampling level.
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ed/core/config.h>

#if defined(__CUDACC__)
#define ED_SLC_HD __host__ __device__ __forceinline__
#else
#define ED_SLC_HD inline
#endif

namespace ed::symmetry {

/// The most blocks a code has (its block size L is at least N / 8).
inline constexpr int kSublatticeMaxBlocks = 8;

/// What the canonicalisers read, on the host or the device (the device mirror points it at device
/// copies). A null `to_key` means no code: the plain scan over the group.
struct SublatticeView {
    const std::uint64_t* to_key   = nullptr;   // bpw x 256: a state's bits moved to their key positions
    const std::uint16_t* lead     = nullptr;   // m x 2^L: the least leading block from block j's pattern
    const std::uint32_t* cand_off = nullptr;   // m x 2^L + 1: offsets into cand
    const std::uint16_t* cand     = nullptr;   // per (j, pattern): the elements reaching lead, ascending
    int n_sites = 0, bpw = 0, L = 0, m = 0;

    [[nodiscard]] ED_SLC_HD bool engaged() const noexcept { return to_key != nullptr; }

    /// The state with its bits in key order.
    [[nodiscard]] ED_SLC_HD std::uint64_t key(std::uint64_t s) const noexcept {
        std::uint64_t k = 0;
        for (int b = 0; b < bpw; ++b) k |= to_key[(b << 8) + static_cast<int>((s >> (8 * b)) & 0xFFu)];
        return k;
    }

    /// Block j's pattern in a key: key bits [N - (j + 1) L, N - j L).
    [[nodiscard]] ED_SLC_HD std::uint32_t pattern(std::uint64_t k, int j) const noexcept {
        return static_cast<std::uint32_t>(k >> (n_sites - (j + 1) * L)) & ((1u << L) - 1u);
    }

    /// The least leading block over the images of the state with key k. The identity keeps the
    /// state's own leading block, pattern(k, 0), so this is at most that; less means the state is
    /// not its orbit's representative.
    [[nodiscard]] ED_SLC_HD std::uint32_t least_lead(std::uint64_t k) const noexcept {
        std::uint32_t best = 0xFFFFFFFFu;
        for (int j = 0; j < m; ++j) {
            const std::uint32_t v = lead[(static_cast<std::uint32_t>(j) << L) + pattern(k, j)];
            if (v < best) best = v;
        }
        return best;
    }

    /// fn(g), in ascending element order, for every element whose image of the state (key k) has
    /// the least leading block: every element that maps the state to its representative (the image
    /// least in the whole key) is among them.
    template <class Fn>
    ED_SLC_HD void for_each_candidate(std::uint64_t k, Fn&& fn) const {
        std::uint32_t entry[kSublatticeMaxBlocks];
        std::uint32_t best = 0xFFFFFFFFu;
        for (int j = 0; j < m; ++j) {
            entry[j] = (static_cast<std::uint32_t>(j) << L) + pattern(k, j);
            if (lead[entry[j]] < best) best = lead[entry[j]];
        }
        std::uint32_t pos[kSublatticeMaxBlocks], end[kSublatticeMaxBlocks];
        int n = 0;
        for (int j = 0; j < m; ++j)
            if (lead[entry[j]] == best) {
                pos[n] = cand_off[entry[j]];
                end[n] = cand_off[entry[j] + 1];
                ++n;
            }
        if (n == 1) {   // the common case: one block reaches the least leading block
            for (std::uint32_t p = pos[0]; p < end[0]; ++p) fn(static_cast<int>(cand[p]));
            return;
        }
        for (;;) {   // merge the blocks' ascending lists (each element maps one block to the lead)
            int pick = -1;
            std::uint32_t g = 0xFFFFFFFFu;
            for (int i = 0; i < n; ++i)
                if (pos[i] < end[i] && cand[pos[i]] < g) { g = cand[pos[i]]; pick = i; }
            if (pick < 0) return;
            ++pos[pick];
            fn(static_cast<int>(g));
        }
    }
};

/// Whether the verb that builds the next orbit tables takes the relaxed rule (1): unset,
/// ED_SYM_SUBLATTICE engages from 16 distinct permutations under it and from 64 otherwise. Only a
/// host sparse apply pays for key-least representatives (in cache locality, see above), so a verb
/// whose blocks run on the device, and one that applies H a few times per block (eigs, spectrum,
/// exact thermodynamics), takes the relaxed rule; host sampled thermodynamics and host dynamics,
/// thousands of applies, the strict one. A verb sets it for its own duration (SublatticeRuleScope);
/// tables and sectors keep the rule they were built with, so a hint that changes later never
/// mixes rules.
inline std::atomic<int>& sublattice_relaxed_hint() noexcept {
    static std::atomic<int> hint{0};
    return hint;
}

class SublatticeRuleScope {
public:
    explicit SublatticeRuleScope(bool relaxed) noexcept : prev_(sublattice_relaxed_hint().exchange(relaxed ? 1 : 0)) {}
    ~SublatticeRuleScope() { sublattice_relaxed_hint().store(prev_); }
    SublatticeRuleScope(const SublatticeRuleScope&) = delete;
    SublatticeRuleScope& operator=(const SublatticeRuleScope&) = delete;

private:
    int prev_;
};

class SublatticeCode {
public:
    /// The code of the group whose elements are `perms` (row-major |G| x N: image bit i is source
    /// bit perms[g * N + i]) with XOR masks `flips` (null: none), or null: switched off, or no block
    /// system of 2..8 blocks of at most 16 sites. Cached by the element list; never freed (a view
    /// into a code stays valid).
    [[nodiscard]] static std::shared_ptr<const SublatticeCode>
    of(const int* perms, const std::uint64_t* flips, int G, int N) {
        return of(perms, flips, G, N, ed::env::tristate("ED_SYM_SUBLATTICE"));
    }

    /// The same with the mode given (nullopt: the size rule; true: whenever a block system exists;
    /// false: never) -- a saved sector rebuilds the code it was computed with whatever the
    /// environment says now.
    [[nodiscard]] static std::shared_ptr<const SublatticeCode>
    of(const int* perms, const std::uint64_t* flips, int G, int N, std::optional<bool> mode) {
        if (G <= 1 || N < 4 || N > 64 || G > 65535 || (mode && !*mode)) return nullptr;
        // the rule: forced (1), or unset with the least number of distinct permutations (-16, -64)
        const int rule = mode ? 1 : -(sublattice_relaxed_hint().load() ? 16 : 64);
        Key key{N, rule,
                std::vector<int>(perms, perms + static_cast<std::size_t>(G) * static_cast<std::size_t>(N)),
                flips ? std::vector<std::uint64_t>(flips, flips + G)
                      : std::vector<std::uint64_t>(static_cast<std::size_t>(G), 0ULL)};
        static std::mutex mu;
        static std::unordered_map<std::uint64_t,
                                  std::vector<std::pair<Key, std::shared_ptr<const SublatticeCode>>>> cache;
        const std::uint64_t h = key.hash();
        std::lock_guard<std::mutex> lk(mu);
        auto& bucket = cache[h];
        for (const auto& e : bucket)
            if (e.first == key) return e.second;
        auto code = build_(key, mode.has_value() ? 0 : -rule);
        bucket.emplace_back(std::move(key), code);
        return code;
    }

    /// The view of the code of a group, or an empty view (no code).
    [[nodiscard]] static SublatticeView view_of(const int* perms, const std::uint64_t* flips, int G, int N) {
        const auto c = of(perms, flips, G, N);
        return c ? c->view() : SublatticeView{};
    }

    [[nodiscard]] SublatticeView view() const noexcept {
        return {to_key_.data(), lead_.data(), cand_off_.data(), cand_.data(), n_, bpw_, L_, m_};
    }

    [[nodiscard]] int block_size() const noexcept { return L_; }
    [[nodiscard]] int blocks() const noexcept { return m_; }
    /// The key position of every site.
    [[nodiscard]] const std::vector<int>& key_bit() const noexcept { return key_bit_; }
    /// Mixed into the content hash of what depends on the representatives (orbit tables).
    [[nodiscard]] std::uint64_t fingerprint() const noexcept { return fingerprint_; }
    /// The mean number of candidates over the (block, pattern) entries.
    [[nodiscard]] double mean_candidates() const noexcept {
        return lead_.empty() ? 0.0 : static_cast<double>(cand_.size()) / static_cast<double>(lead_.size());
    }
    [[nodiscard]] const std::vector<std::uint64_t>& to_key() const noexcept { return to_key_; }
    [[nodiscard]] const std::vector<std::uint16_t>& lead() const noexcept { return lead_; }
    [[nodiscard]] const std::vector<std::uint32_t>& cand_off() const noexcept { return cand_off_; }
    [[nodiscard]] const std::vector<std::uint16_t>& cand() const noexcept { return cand_; }

private:
    struct Key {
        int N = 0, mode = -1;
        std::vector<int> perms;
        std::vector<std::uint64_t> flips;
        bool operator==(const Key& o) const {
            return N == o.N && mode == o.mode && perms == o.perms && flips == o.flips;
        }
        [[nodiscard]] std::uint64_t hash() const {
            std::uint64_t h = 1469598103934665603ULL;
            auto mix = [&h](std::uint64_t v) { h ^= v; h *= 1099511628211ULL; };
            mix(static_cast<std::uint64_t>(N));
            mix(static_cast<std::uint64_t>(mode + 2));
            for (int p : perms) mix(static_cast<std::uint64_t>(p));
            for (std::uint64_t f : flips) mix(f);
            return h;
        }
    };

    std::vector<std::uint64_t> to_key_;
    std::vector<std::uint16_t> lead_;
    std::vector<std::uint32_t> cand_off_;
    std::vector<std::uint16_t> cand_;
    std::vector<int>           key_bit_;
    int n_ = 0, bpw_ = 0, L_ = 0, m_ = 0;
    std::uint64_t fingerprint_ = 0;

    /// The block system with the largest blocks (at most 16 sites, 2..8 blocks of equal size), as
    /// each site's block, blocks numbered by their least site; empty when the group permutes none.
    /// Candidates: for every site b, the finest partition invariant under every element that puts
    /// sites 0 and b together (union-find closed under the elements).
    [[nodiscard]] static std::vector<int> block_system_(const std::vector<std::vector<int>>& sp, int N, int& L) {
        L = 0;
        std::vector<int> best;
        std::vector<int> parent(static_cast<std::size_t>(N));
        auto find = [&parent](int x) {
            while (parent[static_cast<std::size_t>(x)] != x) {
                parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
                x = parent[static_cast<std::size_t>(x)];
            }
            return x;
        };
        for (int b = 1; b < N; ++b) {
            std::iota(parent.begin(), parent.end(), 0);
            parent[static_cast<std::size_t>(b)] = 0;
            std::vector<std::pair<int, int>> todo{{0, b}};
            while (!todo.empty()) {
                const auto [x, y] = todo.back();
                todo.pop_back();
                for (const auto& p : sp) {
                    const int px = p[static_cast<std::size_t>(x)], py = p[static_cast<std::size_t>(y)];
                    const int u = find(px), v = find(py);
                    if (u == v) continue;
                    parent[static_cast<std::size_t>(std::max(u, v))] = std::min(u, v);
                    todo.emplace_back(px, py);
                }
            }
            std::vector<int> size(static_cast<std::size_t>(N), 0);
            for (int s = 0; s < N; ++s) ++size[static_cast<std::size_t>(find(s))];
            const int l = size[0];
            if (l > 16 || l <= L || N % l != 0 || N / l < 2 || N / l > kSublatticeMaxBlocks) continue;
            bool equal = true;
            for (int s = 0; s < N && equal; ++s) equal = size[static_cast<std::size_t>(find(s))] == l;
            if (!equal) continue;
            std::vector<int> root_block(static_cast<std::size_t>(N), -1), block(static_cast<std::size_t>(N));
            int nb = 0;
            for (int s = 0; s < N; ++s) {
                int& rb = root_block[static_cast<std::size_t>(find(s))];
                if (rb < 0) rb = nb++;
                block[static_cast<std::size_t>(s)] = rb;
            }
            L = l;
            best = std::move(block);
        }
        return best;
    }

    // min_perms 0: forced; else the size rule (N >= 24 and at least min_perms distinct permutations).
    [[nodiscard]] static std::shared_ptr<const SublatticeCode> build_(const Key& k, std::size_t min_perms) {
        const int N = k.N;
        const int G = static_cast<int>(k.flips.size());
        // The distinct site permutations decide whether and how (a flip half repeats them).
        std::vector<std::vector<int>> sp;
        for (int g = 0; g < G; ++g) {
            std::vector<int> p(k.perms.begin() + static_cast<std::ptrdiff_t>(g) * N,
                               k.perms.begin() + static_cast<std::ptrdiff_t>(g + 1) * N);
            if (std::find(sp.begin(), sp.end(), p) == sp.end()) sp.push_back(std::move(p));
        }
        if (min_perms > 0 && (N < 24 || sp.size() < min_perms)) return nullptr;
        int L = 0;
        const std::vector<int> block = block_system_(sp, N, L);
        if (block.empty()) return nullptr;
        auto c = std::make_shared<SublatticeCode>();
        c->n_ = N;
        c->L_ = L;
        c->m_ = N / L;
        c->bpw_ = (N + 7) / 8;
        // Key order: block j holds key bits [N - (j + 1) L, N - j L), its sites ascending.
        c->key_bit_.assign(static_cast<std::size_t>(N), 0);
        std::vector<std::vector<int>> sites(static_cast<std::size_t>(c->m_));
        for (int s = 0; s < N; ++s) sites[static_cast<std::size_t>(block[static_cast<std::size_t>(s)])].push_back(s);
        for (int j = 0; j < c->m_; ++j)
            for (int t = 0; t < L; ++t)
                c->key_bit_[static_cast<std::size_t>(sites[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)])] =
                    N - (j + 1) * L + t;
        std::uint64_t h = 1469598103934665603ULL;
        for (int kb : c->key_bit_) { h ^= static_cast<std::uint64_t>(kb + 1); h *= 1099511628211ULL; }
        c->fingerprint_ = h;
        c->to_key_.assign(static_cast<std::size_t>(c->bpw_) * 256, 0ULL);
        for (int b = 0; b < c->bpw_; ++b)
            for (int v = 0; v < 256; ++v) {
                std::uint64_t out = 0;
                for (int t = 0; t < 8 && 8 * b + t < N; ++t)
                    if ((v >> t) & 1) out |= std::uint64_t{1} << c->key_bit_[static_cast<std::size_t>(8 * b + t)];
                c->to_key_[static_cast<std::size_t>(b) * 256 + static_cast<std::size_t>(v)] = out;
            }
        // Leading-block tables. Element g fills key bit N - L + t (block 0's t-th site i) from source
        // site perms[g][i]; those sources make up one block j(g), so the leading block of g's image of
        // a state whose block j(g) holds pattern x is the OR of e_g[u] over x's set bits u, XOR the
        // leading bits of g's flip.
        const std::size_t P = std::size_t{1} << L;
        const std::uint32_t lmask = (1u << L) - 1u;
        std::vector<int> src_block(static_cast<std::size_t>(G));
        std::vector<std::vector<std::uint32_t>> e(static_cast<std::size_t>(G),
                                                  std::vector<std::uint32_t>(static_cast<std::size_t>(L), 0u));
        std::vector<std::uint32_t> fl(static_cast<std::size_t>(G), 0u);
        std::vector<int> pos_in_block(static_cast<std::size_t>(N), -1);
        for (int j = 0; j < c->m_; ++j)
            for (int t = 0; t < L; ++t)
                pos_in_block[static_cast<std::size_t>(sites[static_cast<std::size_t>(j)][static_cast<std::size_t>(t)])] = t;
        for (int g = 0; g < G; ++g) {
            const int* p = k.perms.data() + static_cast<std::ptrdiff_t>(g) * N;
            src_block[static_cast<std::size_t>(g)] = block[static_cast<std::size_t>(p[sites[0][0]])];
            for (int t = 0; t < L; ++t) {
                const int i = sites[0][static_cast<std::size_t>(t)];
                e[static_cast<std::size_t>(g)][static_cast<std::size_t>(pos_in_block[static_cast<std::size_t>(p[i])])] =
                    1u << t;
                if ((k.flips[static_cast<std::size_t>(g)] >> i) & 1ULL) fl[static_cast<std::size_t>(g)] |= 1u << t;
            }
        }
        const std::size_t M = static_cast<std::size_t>(c->m_) * P;
        c->lead_.assign(M, 0xFFFFu);
        std::vector<std::uint32_t> img(P), count(M, 0);
        auto images = [&](int g) {   // the leading block for every pattern of g's source block
            const auto& eg = e[static_cast<std::size_t>(g)];
            img[0] = 0;
            for (std::size_t x = 1; x < P; ++x)
                img[x] = img[x & (x - 1)] | eg[static_cast<std::size_t>(__builtin_ctzll(static_cast<unsigned long long>(x)))];
            for (std::size_t x = 0; x < P; ++x) img[x] = (img[x] ^ fl[static_cast<std::size_t>(g)]) & lmask;
        };
        for (int g = 0; g < G; ++g) {
            images(g);
            std::uint16_t* row = c->lead_.data() + static_cast<std::size_t>(src_block[static_cast<std::size_t>(g)]) * P;
            for (std::size_t x = 0; x < P; ++x)
                if (img[x] < row[x]) row[x] = static_cast<std::uint16_t>(img[x]);
        }
        for (int g = 0; g < G; ++g) {
            images(g);
            const std::size_t base = static_cast<std::size_t>(src_block[static_cast<std::size_t>(g)]) * P;
            for (std::size_t x = 0; x < P; ++x) count[base + x] += c->lead_[base + x] == img[x] ? 1u : 0u;
        }
        c->cand_off_.assign(M + 1, 0);
        for (std::size_t i = 0; i < M; ++i) c->cand_off_[i + 1] = c->cand_off_[i] + count[i];
        c->cand_.assign(c->cand_off_[M], 0);
        std::vector<std::uint32_t> fill(c->cand_off_.begin(), c->cand_off_.end() - 1);
        for (int g = 0; g < G; ++g) {   // ascending g, so each list ascends
            images(g);
            const std::size_t base = static_cast<std::size_t>(src_block[static_cast<std::size_t>(g)]) * P;
            for (std::size_t x = 0; x < P; ++x)
                if (c->lead_[base + x] == img[x]) c->cand_[fill[base + x]++] = static_cast<std::uint16_t>(g);
        }
        return c;
    }
};

}  // namespace ed::symmetry
