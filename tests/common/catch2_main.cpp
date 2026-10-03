// =============================================================================
// tests/common/catch2_main.cpp -- the unit tests' main: Catch2's session, with an exit code ctest can
// trust.
//
// Catch2 exits with the number of failed assertions (at most 255) when something fails and with 4 when
// every test case it ran was skipped. Under SKIP_RETURN_CODE 4 a run with exactly four failed
// assertions therefore read as a skip, and the gate passed it (test_row_walk, gate 62664486). Here the
// run's totals decide: any failure exits 1, a run whose test cases were all skipped exits 77 (the
// SKIP_RETURN_CODE ed_add_test registers), anything else keeps Catch2's code (0, or a usage error).
// =============================================================================
#include <catch2/catch_session.hpp>
#include <catch2/catch_totals.hpp>
#include <catch2/interfaces/catch_interfaces_reporter.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace {

Catch::Totals g_totals;

class Totals final : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;
    void testRunEnded(Catch::TestRunStats const& stats) override { g_totals = stats.totals; }
};

}  // namespace

CATCH_REGISTER_LISTENER(Totals)

int main(int argc, char* argv[]) {
    const int rc = Catch::Session().run(argc, argv);
    if (g_totals.assertions.failed > 0 || g_totals.testCases.failed > 0) return 1;
    if (g_totals.testCases.total() > 0 && g_totals.testCases.skipped == g_totals.testCases.total()) return 77;
    return rc;
}
