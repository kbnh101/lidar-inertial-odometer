#pragma once
// Module timing report shared by the ICP test binaries. Timing is off by default in IcpOptions, so
// a test must opt in with TimedOptions(); the report functions then print where do_icp() spent its
// time. Nothing here asserts on durations -- timings are informational, machine dependent output.
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "p2ptpl_icp/icp_point_to_point_plane.hpp"

namespace icp_test
{
/// "Suite.Test" of the running test, so every table says which test produced it.
inline std::string CurrentTestName()
{
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    return info == nullptr ? std::string("(outside a test)") : std::string(info->test_suite_name()) + '.' + info->name();
}

/// Options with module timing enabled; without it every IcpTiming field stays 0.
inline p2ptpl_icp::IcpOptions TimedOptions(p2ptpl_icp::IcpOptions options = {})
{
    options.collect_timing = true;
    return options;
}

/// Module breakdown of one registration, optionally followed by its per-outer-iteration table.
inline void ReportTiming(const std::string& detail, const p2ptpl_icp::IcpResult& result, bool per_iteration = true)
{
    const std::string label = CurrentTestName() + " [" + detail + "]";
    p2ptpl_icp::PrintIcpTimingRun(std::cout, label.c_str(), result.timing);
    if (per_iteration)
    {
        p2ptpl_icp::PrintIterationTiming(std::cout, result.history, "  per outer iteration");
    }
}

/// Module breakdown aggregated over repeated registrations of the *same* workload.
inline void ReportTimingSamples(const std::string& detail, const std::vector<p2ptpl_icp::IcpTiming>& samples)
{
    const std::string label = CurrentTestName() + " [" + detail + "]";
    p2ptpl_icp::PrintIcpTiming(std::cout, samples, label.c_str());
}
}  // namespace icp_test
