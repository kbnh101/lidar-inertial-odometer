// Checks of the fused point-to-point + point-to-plane residual against the derivation note:
//   1. the worked single-pair example (note (52)-(55)),
//   2. central-difference Jacobians through the same SE(3) retraction (note (56)),
//   3. 3-D fused residual, 4-D stacked residual and direct H/b accumulation agree (note (46), (50)),
//   4. Ceres assembles exactly that tangent-space Jacobian through Se3RightManifold,
//   5. the joint Huber cost matches Ceres' loss (note (61), (64)),
//   6. alpha > 0 removes the tangent-plane null space of pure point-to-plane (note, page 6),
//   7. ICP recovers the ground truth pose from the sample files and from a synthetic scene.

#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "common/point_cloud.hpp"
#include "fused_icp/fused_cost.hpp"
#include "fused_icp/fused_residual.hpp"
#include "fused_icp/icp_fused_point_plane.hpp"
#include "fused_icp/se3.hpp"

namespace
{
using fused_icp::Correspondence;
using fused_icp::FusedWeights;
using fused_icp::Matrix36;
using fused_icp::Matrix66;
using fused_icp::Vector6;

std::string data_path(const std::string& file)
{
    return std::string(FUSED_ICP_DATA_DIR) + "/" + file;
}

Eigen::Isometry3d make_pose(const Eigen::Vector3d& rotation_vector, const Eigen::Vector3d& translation)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = fused_icp::se3::exp_so3(rotation_vector);
    pose.translation() = translation;
    return pose;
}

/// Random pose, point and unit normal generator with the note's seed.
struct RandomScene
{
    std::mt19937 rng{42};
    std::uniform_real_distribution<double> uniform{-1.0, 1.0};

    Eigen::Vector3d vector(double scale)
    {
        return scale * Eigen::Vector3d(uniform(rng), uniform(rng), uniform(rng));
    }

    Eigen::Isometry3d pose()
    {
        return make_pose(vector(1.5), vector(3.0));
    }

    Correspondence correspondence()
    {
        Correspondence pair;
        pair.source_point = vector(5.0);
        pair.target_point = vector(5.0);
        pair.target_normal = vector(1.0).normalized();
        return pair;
    }
};

/// Central difference through the retraction of the note, (56): [r(T Exp(h u_k)) - r(T Exp(-h u_k))] / 2h.
template <typename ResidualFn>
Eigen::MatrixXd numeric_jacobian(const Eigen::Isometry3d& pose, int rows, ResidualFn residual, double h = 1e-6)
{
    Eigen::MatrixXd J(rows, 6);
    for (int k = 0; k < 6; ++k)
    {
        Vector6 step = Vector6::Zero();
        step[k] = h;
        const Eigen::VectorXd plus = residual(fused_icp::se3::retract(pose, step));
        const Eigen::VectorXd minus = residual(fused_icp::se3::retract(pose, -step));
        J.col(k) = (plus - minus) / (2.0 * h);
    }
    return J;
}

const std::vector<FusedWeights> kWeightSets = {{4.0, 5.0}, {0.0, 5.0}, {4.0, 0.0}, {0.03, 12.0}};

Eigen::MatrixXd crs_to_dense(const ceres::CRSMatrix& crs)
{
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(crs.num_rows, crs.num_cols);
    for (int row = 0; row < crs.num_rows; ++row)
    {
        for (int k = crs.rows[row]; k < crs.rows[row + 1]; ++k)
        {
            dense(row, crs.cols[k]) = crs.values[k];
        }
    }
    return dense;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// 1. Worked example of the note, (52)-(55)
// ---------------------------------------------------------------------------------------------
TEST(FusedResidual, WorkedExampleOfTheNote)
{
    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    const Eigen::Vector3d p(1.0, 2.0, 3.0);
    const Eigen::Vector3d q(0.9, 2.2, 2.7);
    const Eigen::Vector3d n(0.0, 0.0, 1.0);
    const FusedWeights weights{4.0, 5.0};

    const Eigen::Vector3d e = fused_icp::geometric_error(pose, p, q);
    EXPECT_TRUE(e.isApprox(Eigen::Vector3d(0.1, -0.2, 0.3), 1e-12));

    const Eigen::Matrix3d omega = fused_icp::information_matrix(n, weights);
    EXPECT_TRUE(omega.isApprox(Eigen::Vector3d(4.0, 4.0, 9.0).asDiagonal().toDenseMatrix(), 1e-12));

    const Eigen::Matrix3d L = fused_icp::fusion_whitener(n, weights);
    EXPECT_TRUE(L.isApprox(Eigen::Vector3d(2.0, 2.0, 3.0).asDiagonal().toDenseMatrix(), 1e-12));
    EXPECT_TRUE((L.transpose() * L).isApprox(omega, 1e-12));  // L^T L = Omega, (39)

    const Eigen::Vector3d r = fused_icp::fused_residual(e, n, weights);
    EXPECT_TRUE(r.isApprox(Eigen::Vector3d(0.2, -0.4, 0.9), 1e-12));

    Matrix36 G_expected;
    // clang-format off
    G_expected <<  0.0,  3.0, -2.0, 1.0, 0.0, 0.0,
                  -3.0,  0.0,  1.0, 0.0, 1.0, 0.0,
                   2.0, -1.0,  0.0, 0.0, 0.0, 1.0;
    // clang-format on
    const Matrix36 G = fused_icp::geometric_jacobian(pose.linear(), p);
    EXPECT_TRUE(G.isApprox(G_expected, 1e-12));

    Matrix36 J_expected;
    // clang-format off
    J_expected <<  0.0,  6.0, -4.0, 2.0, 0.0, 0.0,
                  -6.0,  0.0,  2.0, 0.0, 2.0, 0.0,
                   6.0, -3.0,  0.0, 0.0, 0.0, 3.0;
    // clang-format on
    EXPECT_TRUE(fused_icp::fused_jacobian(G, n, weights).isApprox(J_expected, 1e-12));

    // 2C = alpha |e|^2 + beta (n^T e)^2 = |r_f|^2 = 1.01, (55)
    const double two_cost = weights.alpha * e.squaredNorm() + weights.beta * std::pow(n.dot(e), 2);
    EXPECT_NEAR(two_cost, 1.01, 1e-12);
    EXPECT_NEAR(r.squaredNorm(), 1.01, 1e-12);
}

// ---------------------------------------------------------------------------------------------
// 2. Central differences through the same retraction, (56)
// ---------------------------------------------------------------------------------------------
TEST(FusedResidual, AnalyticJacobiansMatchCentralDifferences)
{
    RandomScene scene;
    double max_error_G = 0.0, max_error_pt = 0.0, max_error_pl = 0.0, max_error_f = 0.0;

    for (int sample = 0; sample < 100; ++sample)
    {
        const Eigen::Isometry3d pose = scene.pose();
        const Correspondence pair = scene.correspondence();
        const FusedWeights weights = kWeightSets[sample % kWeightSets.size()];

        // Geometric Jacobian G, (20)
        const Matrix36 G = fused_icp::geometric_jacobian(pose.linear(), pair.source_point);
        const Eigen::MatrixXd G_num = numeric_jacobian(pose, 3,
                                                       [&](const Eigen::Isometry3d& T) -> Eigen::VectorXd
                                                       {
                                                           return fused_icp::geometric_error(T, pair.source_point, pair.target_point);
                                                       });
        max_error_G = std::max(max_error_G, (G - G_num).cwiseAbs().maxCoeff());

        // Point-to-point J_pt = sqrt(alpha) G, (24)
        const Eigen::MatrixXd J_pt_num =
                numeric_jacobian(pose, 3,
                                 [&](const Eigen::Isometry3d& T) -> Eigen::VectorXd
                                 {
                                     return std::sqrt(weights.alpha) * fused_icp::geometric_error(T, pair.source_point, pair.target_point);
                                 });
        max_error_pt = std::max(max_error_pt, (std::sqrt(weights.alpha) * G - J_pt_num).cwiseAbs().maxCoeff());

        // Point-to-plane J_pl = sqrt(beta) n^T G, (27)
        const Eigen::MatrixXd J_pl_num = numeric_jacobian(
                pose, 1,
                [&](const Eigen::Isometry3d& T) -> Eigen::VectorXd
                {
                    Eigen::VectorXd r(1);
                    r[0] = std::sqrt(weights.beta) * pair.target_normal.dot(fused_icp::geometric_error(T, pair.source_point, pair.target_point));
                    return r;
                });
        const Eigen::MatrixXd J_pl = std::sqrt(weights.beta) * pair.target_normal.transpose() * G;
        max_error_pl = std::max(max_error_pl, (J_pl - J_pl_num).cwiseAbs().maxCoeff());

        // Fused J_f = L G, (44)
        const Matrix36 J_f = fused_icp::fused_jacobian(G, pair.target_normal, weights);
        const Eigen::MatrixXd J_f_num = numeric_jacobian(
                pose, 3,
                [&](const Eigen::Isometry3d& T) -> Eigen::VectorXd
                {
                    return fused_icp::fused_residual(fused_icp::geometric_error(T, pair.source_point, pair.target_point), pair.target_normal, weights);
                });
        max_error_f = std::max(max_error_f, (J_f - J_f_num).cwiseAbs().maxCoeff());

        // The Ceres cost function returns the same J_f in its tangent columns.
        fused_icp::FusedPointPlaneCostFunction cost(pair.source_point, pair.target_point, pair.target_normal, weights);
        const fused_icp::PoseParameters parameters = fused_icp::pose_to_parameters(pose);
        const double* parameter_blocks[] = {parameters.data()};
        Eigen::Vector3d residual;
        Eigen::Matrix<double, 3, fused_icp::kPoseAmbientSize, Eigen::RowMajor> ambient_jacobian;
        double* jacobian_blocks[] = {ambient_jacobian.data()};
        ASSERT_TRUE(cost.Evaluate(parameter_blocks, residual.data(), jacobian_blocks));
        EXPECT_TRUE(ambient_jacobian.leftCols<6>().isApprox(J_f, 1e-12));
        EXPECT_TRUE(ambient_jacobian.col(6).isZero());
    }

    // The note reports ~1e-9 for double precision and h = 1e-6.
    std::printf("max |analytic - central difference|: G %.2e | J_pt %.2e | J_pl %.2e | J_f %.2e\n", max_error_G, max_error_pt, max_error_pl, max_error_f);
    EXPECT_LT(max_error_G, 1e-7);
    EXPECT_LT(max_error_pt, 1e-7);
    EXPECT_LT(max_error_pl, 1e-7);
    EXPECT_LT(max_error_f, 1e-7);
}

// ---------------------------------------------------------------------------------------------
// 3. 3-D fused, 4-D stacked and direct accumulation give the same normal equations, (46), (50)
// ---------------------------------------------------------------------------------------------
TEST(FusedResidual, ThreeDimensionalAndStackedFormsAgree)
{
    RandomScene scene;
    for (const FusedWeights& weights : kWeightSets)
    {
        const Eigen::Isometry3d pose = scene.pose();
        std::vector<Correspondence> pairs;
        for (int i = 0; i < 50; ++i)
        {
            pairs.push_back(scene.correspondence());
        }

        Matrix66 H3 = Matrix66::Zero(), H4 = Matrix66::Zero();
        Vector6 b3 = Vector6::Zero(), b4 = Vector6::Zero();
        for (const Correspondence& pair : pairs)
        {
            const Eigen::Vector3d e = fused_icp::geometric_error(pose, pair.source_point, pair.target_point);
            const Matrix36 G = fused_icp::geometric_jacobian(pose.linear(), pair.source_point);

            // r_3 = L e, J_3 = L G
            const Eigen::Vector3d r3 = fused_icp::fused_residual(e, pair.target_normal, weights);
            const Matrix36 J3 = fused_icp::fused_jacobian(G, pair.target_normal, weights);
            H3 += J3.transpose() * J3;
            b3 -= J3.transpose() * r3;

            // r_4 = [sqrt(alpha) e; sqrt(beta) n^T e], J_4 = [sqrt(alpha) G; sqrt(beta) n^T G]
            Eigen::Vector4d r4;
            r4 << std::sqrt(weights.alpha) * e, std::sqrt(weights.beta) * pair.target_normal.dot(e);
            Eigen::Matrix<double, 4, 6> J4;
            J4 << std::sqrt(weights.alpha) * G, std::sqrt(weights.beta) * pair.target_normal.transpose() * G;
            H4 += J4.transpose() * J4;
            b4 -= J4.transpose() * r4;
        }

        const fused_icp::NormalEquations direct = fused_icp::accumulate_normal_equations(pose, pairs, weights, 0.0);

        const double scale = std::max(1.0, H3.cwiseAbs().maxCoeff());
        EXPECT_LT((H3 - H4).cwiseAbs().maxCoeff() / scale, 1e-13);
        EXPECT_LT((H3 - direct.H).cwiseAbs().maxCoeff() / scale, 1e-13);
        const double b_scale = std::max(1.0, b3.cwiseAbs().maxCoeff());
        EXPECT_LT((b3 - b4).cwiseAbs().maxCoeff() / b_scale, 1e-13);
        EXPECT_LT((b3 - direct.b).cwiseAbs().maxCoeff() / b_scale, 1e-13);
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Ceres + Se3RightManifold assemble exactly J_f (tangent space) and the same H, b
// ---------------------------------------------------------------------------------------------
TEST(FusedCost, CeresTangentJacobianEqualsAnalytic)
{
    RandomScene scene;
    const FusedWeights weights{0.5, 2.0};
    const Eigen::Isometry3d pose = scene.pose();
    std::vector<Correspondence> pairs;
    for (int i = 0; i < 20; ++i)
    {
        pairs.push_back(scene.correspondence());
    }

    fused_icp::PoseParameters parameters = fused_icp::pose_to_parameters(pose);
    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    fused_icp::Se3RightManifold manifold;
    problem.AddParameterBlock(parameters.data(), fused_icp::kPoseAmbientSize, &manifold);
    for (const Correspondence& pair : pairs)
    {
        problem.AddResidualBlock(new fused_icp::FusedPointPlaneCostFunction(pair.source_point, pair.target_point, pair.target_normal, weights), nullptr,
                                 parameters.data());
    }

    double cost = 0.0;
    std::vector<double> residuals;
    ceres::CRSMatrix crs;
    ASSERT_TRUE(problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, nullptr, &crs));
    ASSERT_EQ(crs.num_cols, 6);  // tangent size, the manifold has been applied
    const Eigen::MatrixXd J = crs_to_dense(crs);
    const Eigen::Map<const Eigen::VectorXd> r(residuals.data(), static_cast<Eigen::Index>(residuals.size()));

    Eigen::MatrixXd J_expected(3 * pairs.size(), 6);
    Eigen::VectorXd r_expected(3 * pairs.size());
    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        const Eigen::Vector3d e = fused_icp::geometric_error(pose, pairs[i].source_point, pairs[i].target_point);
        const Matrix36 G = fused_icp::geometric_jacobian(pose.linear(), pairs[i].source_point);
        J_expected.middleRows<3>(3 * i) = fused_icp::fused_jacobian(G, pairs[i].target_normal, weights);
        r_expected.segment<3>(3 * i) = fused_icp::fused_residual(e, pairs[i].target_normal, weights);
    }
    EXPECT_TRUE(J.isApprox(J_expected, 1e-12));
    EXPECT_TRUE(r.isApprox(r_expected, 1e-12));

    const fused_icp::NormalEquations direct = fused_icp::accumulate_normal_equations(pose, pairs, weights, 0.0);
    EXPECT_TRUE((J.transpose() * J).isApprox(direct.H, 1e-12));
    EXPECT_TRUE((-J.transpose() * r).isApprox(direct.b, 1e-12));
    EXPECT_NEAR(cost, direct.cost, 1e-12);
}

TEST(FusedCost, ManifoldPlusIsTheRightPerturbation)
{
    RandomScene scene;
    fused_icp::Se3RightManifold manifold;
    for (int sample = 0; sample < 20; ++sample)
    {
        const Eigen::Isometry3d pose = scene.pose();
        const Vector6 delta = 0.3 * Vector6::Random();
        const fused_icp::PoseParameters x = fused_icp::pose_to_parameters(pose);

        fused_icp::PoseParameters x_plus_delta{};
        ASSERT_TRUE(manifold.Plus(x.data(), delta.data(), x_plus_delta.data()));
        const Eigen::Isometry3d expected = pose * fused_icp::se3::exp_se3(delta);  // T Exp(delta_xi), (1)
        EXPECT_TRUE(fused_icp::parameters_to_pose(x_plus_delta.data()).isApprox(expected, 1e-12));

        Vector6 recovered;
        ASSERT_TRUE(manifold.Minus(x_plus_delta.data(), x.data(), recovered.data()));
        EXPECT_TRUE(recovered.isApprox(delta, 1e-9));
    }

    // Exp and Log are inverse to each other also with the V matrix of (22).
    const Vector6 xi = (Vector6() << 0.4, -0.7, 0.2, 1.0, -2.0, 0.5).finished();
    EXPECT_TRUE(fused_icp::se3::log_se3(fused_icp::se3::exp_se3(xi)).isApprox(xi, 1e-10));
}

// ---------------------------------------------------------------------------------------------
// 5. The joint Huber loss of (61)/(64) is what Ceres applies to the 3-D block
// ---------------------------------------------------------------------------------------------
TEST(FusedCost, JointHuberCostMatchesCeresLoss)
{
    RandomScene scene;
    const FusedWeights weights{1.0, 3.0};
    const double huber_delta = 1.0;
    const Eigen::Isometry3d pose = scene.pose();
    std::vector<Correspondence> pairs;
    for (int i = 0; i < 40; ++i)
    {
        pairs.push_back(scene.correspondence());  // random pairs are far apart, so the loss is active
    }

    fused_icp::PoseParameters parameters = fused_icp::pose_to_parameters(pose);
    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    fused_icp::Se3RightManifold manifold;
    problem.AddParameterBlock(parameters.data(), fused_icp::kPoseAmbientSize, &manifold);
    ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);
    for (const Correspondence& pair : pairs)
    {
        problem.AddResidualBlock(new fused_icp::FusedPointPlaneCostFunction(pair.source_point, pair.target_point, pair.target_normal, weights), loss,
                                 parameters.data());
    }

    double cost = 0.0;
    ASSERT_TRUE(problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr));
    const fused_icp::NormalEquations direct = fused_icp::accumulate_normal_equations(pose, pairs, weights, huber_delta);
    EXPECT_NEAR(cost, direct.cost, 1e-10 * std::max(1.0, cost));

    // At least one pair must be inside and one outside the quadratic zone for this to mean anything.
    int inside = 0, outside = 0;
    for (const Correspondence& pair : pairs)
    {
        const Eigen::Vector3d e = fused_icp::geometric_error(pose, pair.source_point, pair.target_point);
        const double s = fused_icp::fused_residual(e, pair.target_normal, weights).squaredNorm();
        (s <= huber_delta * huber_delta ? inside : outside)++;
    }
    EXPECT_GT(outside, 0);
    (void)inside;
}

// ---------------------------------------------------------------------------------------------
// 6. Point weight removes the tangent-plane null space of a single plane (note, page 6 box)
// ---------------------------------------------------------------------------------------------
TEST(FusedResidual, PointWeightRemovesSinglePlaneDegeneracy)
{
    // Points on z = 0 with n = (0, 0, 1). Pure point-to-plane cannot observe x/y translation and
    // z rotation from one plane; the point term does.
    std::vector<Correspondence> pairs;
    for (int i = -3; i <= 3; ++i)
    {
        for (int j = -3; j <= 3; ++j)
        {
            Correspondence pair;
            pair.source_point = Eigen::Vector3d(i, j, 0.0);
            pair.target_point = Eigen::Vector3d(i, j, 0.0);
            pair.target_normal = Eigen::Vector3d::UnitZ();
            pairs.push_back(pair);
        }
    }
    const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    const auto rank_of = [](const Matrix66& H)
    {
        const Eigen::SelfAdjointEigenSolver<Matrix66> eigen(H);
        const double threshold = 1e-9 * eigen.eigenvalues().maxCoeff();
        return (eigen.eigenvalues().array() > threshold).count();
    };

    EXPECT_EQ(rank_of(fused_icp::accumulate_normal_equations(pose, pairs, FusedWeights{0.0, 1.0}, 0.0).H), 3);
    EXPECT_EQ(rank_of(fused_icp::accumulate_normal_equations(pose, pairs, FusedWeights{0.1, 1.0}, 0.0).H), 6);

    // Limits: beta = 0 is sqrt(alpha) e, alpha = 0 is sqrt(beta) n (n^T e), (page 14).
    const Eigen::Vector3d e(0.3, -0.2, 0.5);
    const Eigen::Vector3d n = Eigen::Vector3d(1.0, 1.0, 1.0).normalized();
    EXPECT_TRUE(fused_icp::fused_residual(e, n, FusedWeights{4.0, 0.0}).isApprox(2.0 * e, 1e-12));
    EXPECT_TRUE(fused_icp::fused_residual(e, n, FusedWeights{0.0, 9.0}).isApprox(3.0 * n * n.dot(e), 1e-12));
}

// ---------------------------------------------------------------------------------------------
// 7. ICP end to end
// ---------------------------------------------------------------------------------------------
TEST(IcpFusedPointPlane, RecoversGroundtruthTransformFromFiles)
{
    const common::PointCloud source_points = common::load_point_cloud(data_path("source_point.txt"));
    const common::PointCloud target_points = common::load_point_cloud(data_path("target_point.txt"));

    // target = source + (-0.2, -0.2, 0) exactly, so every weight set shares the same zero-cost pose.
    // Pure point-to-point (beta = 0) started at the identity stops in the same local minimum as the
    // point-to-point-icp package (rms 0.057 m), hence it is only run from a nearby initial guess.
    struct Case
    {
        FusedWeights weights;
        Eigen::Vector3d initial_translation;
    };
    const std::vector<Case> cases = {{FusedWeights{1.0, 1.0}, Eigen::Vector3d::Zero()},
                                     {FusedWeights{0.1, 1.0}, Eigen::Vector3d::Zero()},
                                     {FusedWeights{0.0, 1.0}, Eigen::Vector3d::Zero()},
                                     {FusedWeights{1.0, 0.0}, Eigen::Vector3d(-0.15, -0.15, 0.0)}};
    for (const Case& test_case : cases)
    {
        fused_icp::IcpOptions options;
        options.point_weight = test_case.weights.alpha;
        options.plane_weight = test_case.weights.beta;
        fused_icp::IcpFusedPointPlane icp(options);
        Eigen::Isometry3d initial_guess = Eigen::Isometry3d::Identity();
        initial_guess.translation() = test_case.initial_translation;
        const fused_icp::IcpResult result = icp.do_icp(source_points, target_points, initial_guess);

        constexpr double kEpsilon = 1e-6;
        const std::string label = "alpha=" + std::to_string(test_case.weights.alpha) + " beta=" + std::to_string(test_case.weights.beta);
        EXPECT_TRUE(result.converged) << label;
        EXPECT_NEAR(result.transform.translation().x(), -0.2, kEpsilon) << label;
        EXPECT_NEAR(result.transform.translation().y(), -0.2, kEpsilon) << label;
        EXPECT_NEAR(result.transform.translation().z(), 0.0, kEpsilon) << label;
        EXPECT_NEAR(Eigen::AngleAxisd(result.transform.linear()).angle(), 0.0, kEpsilon) << label;
        EXPECT_NEAR(result.final_error, 0.0, kEpsilon) << label;
        EXPECT_NEAR(result.final_point_error, 0.0, kEpsilon) << label;
        EXPECT_NEAR(result.final_plane_error, 0.0, kEpsilon) << label;
    }
}

TEST(IcpFusedPointPlane, RecoversRotationAndTranslationOnSyntheticCorner)
{
    // Three orthogonal walls sampled on a grid: the same physical samples appear in both clouds,
    // so both the point and the plane term have an exact zero.
    common::PointCloud target;
    for (int i = 0; i <= 20; ++i)
    {
        for (int j = 0; j <= 20; ++j)
        {
            const double u = 0.25 * i, v = 0.25 * j;
            target.push_back({Eigen::Vector3d(u, v, 0.0), Eigen::Vector3d::UnitZ()});
            target.push_back({Eigen::Vector3d(0.0, u, v), Eigen::Vector3d::UnitX()});
            target.push_back({Eigen::Vector3d(u, 0.0, v), Eigen::Vector3d::UnitY()});
        }
    }
    const Eigen::Isometry3d truth = make_pose(Eigen::Vector3d(0.05, -0.08, 0.12), Eigen::Vector3d(0.3, -0.2, 0.15));
    // source = truth^-1 * target, so that truth * source = target.
    const Eigen::Isometry3d inverse = truth.inverse();
    const common::PointCloud source = common::transform_point_cloud(target, inverse.linear(), inverse.translation());

    fused_icp::IcpOptions options;
    options.point_weight = 0.2;
    options.plane_weight = 1.0;
    options.max_correspondence_distance = 1.0;
    options.min_normal_dot = 0.5;
    fused_icp::IcpFusedPointPlane icp(options);
    const fused_icp::IcpResult result = icp.do_icp(source, target);

    EXPECT_TRUE(result.converged);
    const Eigen::Isometry3d error = truth.inverse() * result.transform;
    EXPECT_LT(error.translation().norm(), 1e-6);
    EXPECT_LT(Eigen::AngleAxisd(error.linear()).angle(), 1e-6);
    EXPECT_LT(result.final_error, 1e-6);
    EXPECT_LT(result.final_point_error, 1e-6);
}
