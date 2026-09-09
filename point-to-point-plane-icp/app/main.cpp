// TXT matching demo and wall-clock benchmark. File I/O, target-tree setup and
// the ICP loop are measured separately; printing/evaluation is outside ICP timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/rotation.hpp"
#include "p2ptpl_icp/icp_point_to_point_plane.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Arguments
{
    std::filesystem::path data_dir = P2PTPL_ICP_DATA_DIR;
    int repeat = 10;
    int warmup = 1;
    p2ptpl_icp::IcpOptions icp;
    Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d expected_pose = Eigen::Isometry3d::Identity();
    bool has_expected_pose = false;
};

void Usage(const char* program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "  --data-dir DIR          source_point.txt / target_point.txt (default: bundled data)\n"
              << "  --repeat N              measured independent registrations (default: 10)\n"
              << "  --warmup N              unmeasured registrations (default: 1)\n"
              << "  --backend cpu|cuda      residual/Jacobian backend (default: cuda in CUDA builds)\n"
              << "  --point-weight W        positive point cost weight (default: 1)\n"
              << "  --plane-weight W        positive plane cost weight (default: 1)\n"
              << "  --max-distance M        correspondence distance threshold (default: 1 m)\n"
              << "  --max-iterations N      outer ICP limit (default: 50)\n"
              << "  --initial-pose TX TY TZ ROLL PITCH YAW   initial source -> target pose\n"
              << "  --expected-pose TX TY TZ ROLL PITCH YAW  optional reference for pose errors\n"
              << "  --help                  show this help\n"
              << "Pose translations are metres; angles are radians, R = Rz(yaw) Ry(pitch) Rx(roll).\n"
              << "Exit codes: 0 all measured runs converged, 1 input/error, 2 registration did not converge.\n";
}

double Number(const std::string& text)
{
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value))
    {
        throw std::invalid_argument("expected a finite number: " + text);
    }
    return value;
}

int Count(const std::string& text, int minimum)
{
    const double value = Number(text);
    if (value < minimum || value > std::numeric_limits<int>::max() || std::floor(value) != value)
    {
        throw std::invalid_argument("invalid repeat/iteration count: " + text);
    }
    return static_cast<int>(value);
}

Arguments ParseArguments(int argc, char** argv)
{
    Arguments args;
    // Installed ROS 2 executables live in <prefix>/lib/<package>/. Prefer their
    // bundled data, so moving the install tree does not require the source checkout.
    std::error_code error;
    const auto executable = std::filesystem::canonical(argv[0], error);
    if (!error)
    {
        const auto installed = executable.parent_path().parent_path().parent_path() / "share/point_to_point_plane_icp/data";
        if (std::filesystem::exists(installed / "source_point.txt"))
        {
            args.data_dir = installed;
        }
    }
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        auto next = [&]() -> std::string
        {
            if (++i >= argc)
            {
                throw std::invalid_argument("missing value for " + option);
            }
            return argv[i];
        };
        if (option == "--data-dir")
        {
            args.data_dir = next();
        }
        else if (option == "--repeat")
        {
            args.repeat = Count(next(), 1);
        }
        else if (option == "--warmup")
        {
            args.warmup = Count(next(), 0);
        }
        else if (option == "--backend")
        {
            const auto backend = next();
            if (backend != "cpu" && backend != "cuda")
                throw std::invalid_argument("backend must be cpu or cuda");
            args.icp.use_cuda = backend == "cuda";
        }
        else if (option == "--point-weight")
        {
            args.icp.point_weight = Number(next());
        }
        else if (option == "--plane-weight")
        {
            args.icp.plane_weight = Number(next());
        }
        else if (option == "--max-distance")
        {
            args.icp.max_correspondence_distance = Number(next());
        }
        else if (option == "--max-iterations")
        {
            args.icp.max_iterations = Count(next(), 1);
        }
        else if (option == "--initial-pose" || option == "--expected-pose")
        {
            double values[6];
            for (double& value : values)
            {
                value = Number(next());
            }
            Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
            pose.translation() = Eigen::Vector3d(values[0], values[1], values[2]);
            pose.linear() = common::euler_zyx_to_rotation(values[3], values[4], values[5]);
            if (option == "--initial-pose")
            {
                args.initial_pose = pose;
            }
            else
            {
                args.expected_pose = pose;
                args.has_expected_pose = true;
            }
        }
        else
        {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (args.icp.point_weight <= 0 || args.icp.plane_weight <= 0 || args.icp.max_correspondence_distance <= 0)
    {
        throw std::invalid_argument("weights and max-distance must be positive");
    }
    return args;
}

void PrintTiming(const char* label, std::vector<double> samples)
{
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[middle] : (samples[middle - 1] + samples[middle]) * 0.5;
    const std::size_t p95 = static_cast<std::size_t>(std::ceil(0.95 * samples.size())) - 1;
    std::cout << std::left << std::setw(22) << label << std::right << std::fixed << std::setprecision(6)
              << " mean=" << mean << "  median=" << median << "  p95=" << samples[p95]
              << "  min=" << samples.front() << "  max=" << samples.back() << " ms\n";
}

// Independent post-run diagnostics on the same distance/normal-gated nearest
// neighbours as the solver. This is deliberately outside the timed registration.
void PrintResiduals(const p2ptpl_icp::IcpPointToPointPlane& icp, const Eigen::Isometry3d& pose)
{
    common::KdTree3d tree;
    tree.build(icp.target());
    const auto& options = icp.options();
    double point_squared = 0.0;
    double plane_squared = 0.0;
    std::size_t count = 0;
    for (const auto& source : icp.source())
    {
        const Eigen::Vector3d transformed = pose * source.point;
        int index = -1;
        double distance_squared = 0.0;
        if (!tree.nearest(transformed, &index, &distance_squared) ||
            distance_squared > options.max_correspondence_distance * options.max_correspondence_distance)
        {
            continue;
        }
        const auto& target = icp.target()[index];
        if (target.normal.squaredNorm() <= 0.0 ||
            (options.min_normal_dot > -1.0 && source.normal.squaredNorm() > 0.0 &&
             (pose.linear() * source.normal).dot(target.normal) < options.min_normal_dot))
        {
            continue;
        }
        const Eigen::Vector3d error = transformed - target.point;
        point_squared += error.squaredNorm();
        const double plane_error = target.normal.dot(error);
        plane_squared += plane_error * plane_error;
        ++count;
    }
    const double infinity = std::numeric_limits<double>::infinity();
    std::cout << "point RMS ||e|| [m]   : " << (count ? std::sqrt(point_squared / count) : infinity) << '\n'
              << "plane RMS n^T e [m]  : " << (count ? std::sqrt(plane_squared / count) : infinity) << '\n';
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--help")
            {
                Usage(argv[0]);
                return 0;
            }
        }
        const Arguments args = ParseArguments(argc, argv);
        const auto load_begin = Clock::now();
        const auto source = common::load_point_cloud((args.data_dir / "source_point.txt").string());
        const auto target = common::load_point_cloud((args.data_dir / "target_point.txt").string());
        const double load_ms = Milliseconds(load_begin, Clock::now());
        if (source.empty() || target.empty())
        {
            throw std::runtime_error("source and target clouds must both be nonempty");
        }
        std::cout << "Hybrid ICP: " << source.size() << " source / " << target.size() << " target points\n"
                  << "data directory: " << std::filesystem::absolute(args.data_dir) << '\n'
                  << "build: " << P2PTPL_BUILD_TYPE << " (use Release for timing)\n"
                  << "residual/Jacobian backend: " << (args.icp.use_cuda ? "cuda" : "cpu") << "; Ceres DENSE_QR: CPU\n"
                  << "weights: point=" << args.icp.point_weight << ", plane=" << args.icp.plane_weight << '\n'
                  << "max distance: " << args.icp.max_correspondence_distance << " m; max iterations: " << args.icp.max_iterations << '\n'
                  << "initial source -> target pose:\n" << args.initial_pose.matrix() << '\n'
                  << "warmup: " << args.warmup << "; measured runs: " << args.repeat << '\n';

        auto warmup = [&]()
        {
            p2ptpl_icp::IcpPointToPointPlane icp(args.icp);
            icp.set_source(source);
            icp.set_target(target);
            icp.do_icp(args.initial_pose);
        };
        for (int i = 0; i < args.warmup; ++i)
        {
            warmup();
        }

        std::vector<double> setup_times, icp_times, total_times;
        setup_times.reserve(args.repeat);
        icp_times.reserve(args.repeat);
        total_times.reserve(args.repeat);
        p2ptpl_icp::IcpResult result;
        int converged = 0;
        int min_iterations = std::numeric_limits<int>::max();
        int max_iterations = 0;
        double worst_error = 0.0;
        for (int i = 0; i < args.repeat; ++i)
        {
            // Every run starts with a fresh matcher and the SAME initial pose.
            // No previous solution or target tree is reused to shorten the benchmark.
            const auto begin = Clock::now();
            p2ptpl_icp::IcpPointToPointPlane icp(args.icp);
            icp.set_source(source);
            icp.set_target(target);
            const auto setup_end = Clock::now();
            auto current = icp.do_icp(args.initial_pose);
            const auto end = Clock::now();
            setup_times.push_back(Milliseconds(begin, setup_end));
            icp_times.push_back(Milliseconds(setup_end, end));
            total_times.push_back(Milliseconds(begin, end));
            converged += current.converged && current.iterations > 0 && std::isfinite(current.final_error) && current.transform.matrix().allFinite();
            min_iterations = std::min(min_iterations, current.iterations);
            max_iterations = std::max(max_iterations, current.iterations);
            worst_error = std::isfinite(current.final_error) ? std::max(worst_error, current.final_error) : std::numeric_limits<double>::infinity();
            result = std::move(current);
        }

        std::cout << "\nTiming (steady_clock wall time; warmup excluded)\n"
                  << std::fixed << std::setprecision(6) << "TXT load, once        : " << load_ms << " ms\n";
        PrintTiming("Setup + target kd-tree", setup_times);
        PrintTiming("ICP search + Ceres", icp_times);
        PrintTiming("Registration total", total_times);
        std::cout << "Total excludes file I/O, reporting, final diagnostics and matcher destruction.\n"
                  << "\nConverged runs       : " << converged << '/' << args.repeat << '\n'
                  << "Outer iterations     : " << min_iterations << " .. " << max_iterations << '\n'
                  << std::scientific << "Worst hybrid RMS [m] : " << worst_error << '\n'
                  << "\nLast run result\n"
                  << "Correspondences      : " << result.correspondences << '/' << source.size() << '\n'
                  << "Hybrid RMS [m]       : " << result.final_error << '\n';
        p2ptpl_icp::IcpPointToPointPlane evaluator(args.icp);
        evaluator.set_source(source);
        evaluator.set_target(target);
        PrintResiduals(evaluator, result.transform);
        std::cout << std::fixed << std::setprecision(12) << "source -> target T:\n" << result.transform.matrix() << '\n';
        if (args.has_expected_pose)
        {
            const Eigen::Isometry3d error = args.expected_pose.inverse() * result.transform;
            std::cout << std::scientific << "Translation error [m]: " << error.translation().norm() << '\n'
                      << "Rotation error [rad] : " << Eigen::AngleAxisd(error.linear()).angle() << '\n';
        }
        std::cout << "Convergence is a stopping condition, not proof of the correct alignment.\n";
        return converged == args.repeat ? 0 : 2;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error: " << error.what() << '\n';
        Usage(argv[0]);
        return 1;
    }
}
