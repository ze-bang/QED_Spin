// =============================================================================
// src/api/symmetry_helpers.cpp
//
// `ed::find_symmetries(...)` -- thin wrapper around the named presets in
// `ed/symmetry/group.h`.
// =============================================================================

#include <ed/api/symmetry_helpers.h>

#include <cctype>
#include <stdexcept>
#include <string>

namespace ed {

namespace {

[[nodiscard]] bool ieq(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

}  // namespace

SymmetryGroupInfo
find_symmetries(int n_sites, std::string_view preset) {
    if (preset.empty() || ieq(preset, "translation") || ieq(preset, "u1_translation")) {
        return ed::sym::translation_group_1d(n_sites);
    }
    if (ieq(preset, "translation+reflection") || ieq(preset, "dihedral") ||
        ieq(preset, "D_N") || ieq(preset, "DN")) {
        return ed::sym::translation_group_with_reflection_1d(n_sites);
    }
    if (ieq(preset, "reflection") || ieq(preset, "Z2_reflection") ||
        ieq(preset, "parity")) {
        return ed::sym::group_from_generators(
            n_sites, {ed::sym::reflection_1d(n_sites)});
    }
    if (ieq(preset, "identity") || ieq(preset, "trivial") || ieq(preset, "none")) {
        // Trivial group: only the identity.
        return ed::sym::group_from_generators(
            n_sites, {ed::sym::identity(n_sites)});
    }
    throw std::invalid_argument(
        "ed::find_symmetries: unknown preset '" + std::string{preset} +
        "'. Accepts 'translation', 'translation+reflection', 'reflection', "
        "'identity'.");
}

SymmetryGroupInfo
find_symmetries(int n_sites,
                std::vector<ed::sym::Permutation> generators,
                std::vector<std::vector<int>> sector_quantum_numbers) {
    return ed::sym::group_from_generators(
        n_sites, std::move(generators), std::move(sector_quantum_numbers));
}

}  // namespace ed
