// =============================================================================
// tests/unit/test_env_registry.cpp
//
// The environment registry (include/ed/config/env_registry.h): one meaning of
// "set" for every value type, no duplicate rows, and the unknown-name scan.
// =============================================================================
#include "common/catch2_harness.h"

#include <cstdlib>
#include <set>
#include <string>

#include <ed/config/env_registry.h>

namespace {
struct ScopedEnv {
    std::string name;
    explicit ScopedEnv(const char* n, const char* v) : name(n) { ::setenv(n, v, 1); }
    ~ScopedEnv() { ::unsetenv(name.c_str()); }
};
}  // namespace

TEST_CASE("env registry: rows are unique and well formed", "[env]") {
    std::set<std::string> seen;
    for (const auto& r : ed::env::rows()) {
        const std::string n = r.name;
        REQUIRE((n.rfind("ED_", 0) == 0 || n.rfind("QED_", 0) == 0));
        REQUIRE(seen.insert(n).second);           // no duplicates
        REQUIRE(std::string(r.meaning).size() > 0);
        REQUIRE(std::string(r.scope).size() > 0);
    }
    REQUIRE(seen.size() > 90);
    REQUIRE(ed::env::is_registered("ED_SYM_LG_ONLY_K0"));
    REQUIRE_FALSE(ed::env::is_registered("ED_SYM_LG_ONLY_KO"));
}

TEST_CASE("env registry: a flag set to 0 is OFF, presence alone never enables", "[env]") {
    const char* n = "ED_MEM_GUARD_OFF";
    ::unsetenv(n);
    REQUIRE(ed::env::flag(n, false) == false);
    REQUIRE(ed::env::flag(n, true) == true);
    REQUIRE_FALSE(ed::env::tristate(n).has_value());
    for (const char* off : {"0", "false", "off", "no", "OFF"}) {
        ScopedEnv e(n, off);
        REQUIRE(ed::env::flag(n, true) == false);
        REQUIRE(ed::env::tristate(n).value() == false);
    }
    for (const char* on : {"1", "true", "yes", "2"}) {
        ScopedEnv e(n, on);
        REQUIRE(ed::env::flag(n, false) == true);
    }
    {
        ScopedEnv e(n, "");                        // empty == unset
        REQUIRE(ed::env::flag(n, false) == false);
        REQUIRE(ed::env::flag(n, true) == true);
        REQUIRE_FALSE(ed::env::tristate(n).has_value());
    }
}

TEST_CASE("env registry: numbers fall back to the default when unset, empty or unparsable", "[env]") {
    const char* n = "ED_SYM_LG_GS_RESTARTS";
    ::unsetenv(n);
    REQUIRE(ed::env::integer(n, 4) == 4);
    { ScopedEnv e(n, "");     REQUIRE(ed::env::integer(n, 4) == 4); }
    { ScopedEnv e(n, "abc");  REQUIRE(ed::env::integer(n, 4) == 4); }
    { ScopedEnv e(n, "30");   REQUIRE(ed::env::integer(n, 4) == 30); }
    { ScopedEnv e(n, "-2");   REQUIRE(ed::env::integer(n, 4) == -2); }
    const char* r = "ED_SYM_LG_GS_RESID_TOL";
    ::unsetenv(r);
    REQUIRE(ed::env::real(r, 1e-8) == 1e-8);
    { ScopedEnv e(r, "1e-6"); REQUIRE(ed::env::real(r, 1e-8) == 1e-6); }
    { ScopedEnv e(r, "x");    REQUIRE(ed::env::real(r, 1e-8) == 1e-8); }
}

TEST_CASE("env registry: snapshot lists what is set; unknown() catches a misspelt name", "[env]") {
    ScopedEnv good("ED_SYM_LG_SEED", "7");
    ScopedEnv typo("ED_SYM_LG_SEEED", "7");
    ScopedEnv harness("ED_TEST_TMP_DIR", "x");
    bool in_snapshot = false;
    for (const auto& kv : ed::env::snapshot())
        in_snapshot = in_snapshot || (kv.first == "ED_SYM_LG_SEED" && kv.second == "7");
    REQUIRE(in_snapshot);
    const auto unk = ed::env::unknown();
    const std::set<std::string> u(unk.begin(), unk.end());
    REQUIRE(u.count("ED_SYM_LG_SEEED") == 1);
    REQUIRE(u.count("ED_SYM_LG_SEED") == 0);
    REQUIRE(u.count("ED_TEST_TMP_DIR") == 0);      // harness namespace is not scanned
    REQUIRE(ed::env::dump("ED_SYM_LG_SEED").find("ED_SYM_LG_SEED") != std::string::npos);
}
