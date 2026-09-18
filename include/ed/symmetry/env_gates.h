#pragma once
// =============================================================================
// include/ed/symmetry/env_gates.h -- the symmetry-stack view of the environment
// registry.
//
// The table of variables lives in <ed/config/env_registry.h> (every ED_* / QED_*
// variable, not only ED_SYM_*). This header keeps the historical entry point:
// ``dump_env_gates()`` renders the ED_SYM_* rows with their live values, and is
// bound to Python as ``_core.dump_env_gates()``.
// =============================================================================

#include <string>

#include <ed/config/env_registry.h>

namespace ed::symmetry {

[[nodiscard]] inline std::string dump_env_gates() { return ed::env::dump("ED_SYM_"); }

}  // namespace ed::symmetry
