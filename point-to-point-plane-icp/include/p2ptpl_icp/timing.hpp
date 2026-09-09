#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <vector>

namespace p2ptpl_icp
{
/// Per-module host wall time of one registration, in milliseconds.
///
/// The *stage* rows partition the registration: each one is measured exactly once, they never
/// overlap, and initialization + ... + diagnostic + other == total. The *detail* rows measure work
/// happening **inside** ceres_solve_ms and overlap each other, so they must never be added to the
/// stages or to one another.
struct IcpTiming
{
    // --- stages: disjoint, add up to total_ms -------------------------------------------------
    double initialization_ms = 0;  ///< input/option checks and Ceres CUDA context creation
    double correspondence_ms = 0;  ///< pose transform + kd-tree nearest neighbour + rejections
    double pivot_ms = 0;  ///< centroid pivot pass over the correspondences
    double problem_setup_ms = 0;  ///< ceres::Problem, parameter block, loss and solver options
    double residual_block_ms = 0;  ///< shared error entry plus two cost blocks per correspondence
    double ceres_solve_ms = 0;  ///< ceres::Solve of one increment subproblem
    double problem_cleanup_ms = 0;  ///< Problem, cost and loss destruction
    double pose_update_ms = 0;  ///< increment composition onto the accumulated pose
    double diagnostic_ms = 0;  ///< evaluate_error around each solve and for the final report
    double other_ms = 0;  ///< total minus every stage above
    double total_ms = 0;

    // --- details inside ceres_solve_ms: overlapping ---------------------------------------------
    double ceres_preprocessor_ms = 0;
    double ceres_minimizer_ms = 0;
    double ceres_postprocessor_ms = 0;
    double ceres_residual_eval_ms = 0;
    double ceres_jacobian_eval_ms = 0;
    double ceres_linear_solver_ms = 0;
    double callback_residual_only_ms = 0;
    double callback_with_jacobian_ms = 0;

    int residual_evaluations = 0;
    int jacobian_evaluations = 0;
    int linear_solves = 0;
    int callback_residual_calls = 0;
    int callback_jacobian_calls = 0;

    double stage_sum_ms() const
    {
        return initialization_ms + correspondence_ms + pivot_ms + problem_setup_ms + residual_block_ms + ceres_solve_ms + problem_cleanup_ms + pose_update_ms +
               diagnostic_ms;
    }
    void finish(double total)
    {
        total_ms = total;
        other_ms = std::max(0.0, total_ms - stage_sum_ms());
    }
};

/// Where a row belongs in the report: a disjoint stage, the total, or an overlapping detail.
enum class TimingSection
{
    kStage,
    kTotal,
    kDetail
};

struct TimingField
{
    const char* name;
    double IcpTiming::*value;
    TimingSection section;
};
inline constexpr std::array<TimingField, 19> kTimingFields{{
    {"initialization_ms", &IcpTiming::initialization_ms, TimingSection::kStage},
    {"correspondence_ms", &IcpTiming::correspondence_ms, TimingSection::kStage},
    {"pivot_ms", &IcpTiming::pivot_ms, TimingSection::kStage},
    {"problem_setup_ms", &IcpTiming::problem_setup_ms, TimingSection::kStage},
    {"residual_block_ms", &IcpTiming::residual_block_ms, TimingSection::kStage},
    {"ceres_solve_ms", &IcpTiming::ceres_solve_ms, TimingSection::kStage},
    {"problem_cleanup_ms", &IcpTiming::problem_cleanup_ms, TimingSection::kStage},
    {"pose_update_ms", &IcpTiming::pose_update_ms, TimingSection::kStage},
    {"diagnostic_ms", &IcpTiming::diagnostic_ms, TimingSection::kStage},
    {"other_ms", &IcpTiming::other_ms, TimingSection::kStage},
    {"total_ms", &IcpTiming::total_ms, TimingSection::kTotal},
    {"ceres_preprocessor_ms", &IcpTiming::ceres_preprocessor_ms, TimingSection::kDetail},
    {"ceres_minimizer_ms", &IcpTiming::ceres_minimizer_ms, TimingSection::kDetail},
    {"ceres_postprocessor_ms", &IcpTiming::ceres_postprocessor_ms, TimingSection::kDetail},
    {"ceres_residual_eval_ms", &IcpTiming::ceres_residual_eval_ms, TimingSection::kDetail},
    {"ceres_jacobian_eval_ms", &IcpTiming::ceres_jacobian_eval_ms, TimingSection::kDetail},
    {"ceres_linear_solver_ms", &IcpTiming::ceres_linear_solver_ms, TimingSection::kDetail},
    {"callback_residual_only_ms", &IcpTiming::callback_residual_only_ms, TimingSection::kDetail},
    {"callback_with_jacobian_ms", &IcpTiming::callback_with_jacobian_ms, TimingSection::kDetail},
}};
struct TimingCount
{
    const char* name;
    int IcpTiming::*value;
};
inline constexpr std::array<TimingCount, 5> kTimingCounts{{
    {"residual_evaluations", &IcpTiming::residual_evaluations},
    {"jacobian_evaluations", &IcpTiming::jacobian_evaluations},
    {"linear_solves", &IcpTiming::linear_solves},
    {"callback_residual_calls", &IcpTiming::callback_residual_calls},
    {"callback_jacobian_calls", &IcpTiming::callback_jacobian_calls},
}};

inline IcpTiming TimingDifference(const IcpTiming& after, const IcpTiming& before)
{
    IcpTiming delta;
    for (const auto& field : kTimingFields)
        delta.*(field.value) = after.*(field.value) - before.*(field.value);
    for (const auto& field : kTimingCounts)
        delta.*(field.value) = after.*(field.value) - before.*(field.value);
    return delta;
}

inline void WriteTimingCsvHeader(std::ostream& out, const char* prefix = "")
{
    for (const auto& field : kTimingFields)
        out << ',' << prefix << field.name;
    for (const auto& field : kTimingCounts)
        out << ',' << prefix << field.name;
}
inline void WriteTimingCsvValues(std::ostream& out, const IcpTiming& timing)
{
    for (const auto& field : kTimingFields)
        out << ',' << timing.*(field.value);
    for (const auto& field : kTimingCounts)
        out << ',' << timing.*(field.value);
}

/// mean / median / p95 / max of one module over several registrations.
struct TimingStatistics
{
    double mean = 0;
    double median = 0;
    double p95 = 0;
    double max = 0;
};

inline TimingStatistics TimingStatisticsOf(const std::vector<IcpTiming>& samples, double IcpTiming::*value)
{
    TimingStatistics statistics;
    if (samples.empty())
        return statistics;
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples)
        values.push_back(sample.*value);
    statistics.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    statistics.median = (values[middle] + values[(values.size() - 1) / 2]) / 2;
    statistics.p95 = values[static_cast<std::size_t>(std::ceil(.95 * values.size())) - 1];
    statistics.max = values.back();
    return statistics;
}

namespace detail
{
inline constexpr const char* kDetailNote =
        "Detail rows below run inside ceres_solve_ms; they overlap, so do NOT add them to the stages or to each other:\n";
inline constexpr const char* kFootnote =
        "Stages are disjoint and add up to total_ms. ceres_linear_solver_ms is Ceres host wall time,\n"
        "including transfers/waits inside its (CPU or CUDA) QR, not CUDA kernel-only time. The callback rows\n"
        "cover shared e / de-dxi cache preparation only, not the complete weighted residual evaluation.\n";

inline void WriteShare(std::ostream& out, double value, double total)
{
    out << std::setprecision(1) << std::setw(7) << (total > 0 ? 100. * value / total : 0.) << "%\n";
}
}  // namespace detail

/// Module breakdown of a single registration: absolute ms and share of its total.
inline void PrintIcpTimingRun(std::ostream& out, const char* label, const IcpTiming& timing)
{
    const auto flags = out.flags();
    const auto precision = out.precision();
    out << '\n' << label << " -- module timing [ms of one registration]\n";
    bool detail_header = false;
    for (const auto& field : kTimingFields)
    {
        if (field.section == TimingSection::kDetail && !detail_header)
        {
            out << detail::kDetailNote;
            detail_header = true;
        }
        out << std::left << std::setw(28) << field.name << std::right << std::fixed << std::setprecision(4) << std::setw(12) << timing.*(field.value);
        detail::WriteShare(out, timing.*(field.value), timing.total_ms);
    }
    out << "calls:";
    for (const auto& field : kTimingCounts)
        out << ' ' << field.name << '=' << timing.*(field.value);
    out << '\n';
    out.flags(flags);
    out.precision(precision);
}

/// Module breakdown aggregated over several registrations; share is the mean share of total_ms.
inline void PrintIcpTiming(std::ostream& out, const std::vector<IcpTiming>& samples, const char* title = "ICP module timing")
{
    if (samples.empty())
        return;
    const auto flags = out.flags();
    const auto precision = out.precision();
    const double total_mean = TimingStatisticsOf(samples, &IcpTiming::total_ms).mean;
    out << '\n' << title << " over " << samples.size() << " registration(s) [ms per registration]\n"
        << std::left << std::setw(28) << "module" << std::right << std::setw(12) << "mean" << std::setw(12) << "median" << std::setw(12) << "p95"
        << std::setw(12) << "max" << std::setw(8) << "share" << '\n';
    bool detail_header = false;
    for (const auto& field : kTimingFields)
    {
        if (field.section == TimingSection::kDetail && !detail_header)
        {
            out << detail::kDetailNote;
            detail_header = true;
        }
        const TimingStatistics statistics = TimingStatisticsOf(samples, field.value);
        out << std::left << std::setw(28) << field.name << std::right << std::fixed << std::setprecision(4) << std::setw(12) << statistics.mean
            << std::setw(12) << statistics.median << std::setw(12) << statistics.p95 << std::setw(12) << statistics.max;
        detail::WriteShare(out, statistics.mean, total_mean);
    }
    out << "Mean calls/registration:";
    for (const auto& field : kTimingCounts)
    {
        double sum = 0;
        for (const auto& sample : samples)
            sum += sample.*(field.value);
        out << ' ' << field.name << '=' << sum / samples.size();
    }
    out << '\n' << detail::kFootnote;
    out.flags(flags);
    out.precision(precision);
}
}  // namespace p2ptpl_icp
