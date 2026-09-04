#include "p2p_icp/rotation.hpp"

#include <cmath>

namespace p2p_icp
{
Eigen::Matrix3d euler_zyx_to_rotation(double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const double cg = std::cos(gamma), sg = std::sin(gamma);

    // R = Rz(gamma) Ry(beta) Rx(alpha), expanded down to the individual entries.
    Eigen::Matrix3d rotation;
    rotation << cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, cb * sg, sa * sb * sg + ca * cg, ca * sb * sg - sa * cg, -sb, sa * cb, ca * cb;
    return rotation;
}

Eigen::Matrix3d euler_zyx_to_rotation(const Eigen::Vector3d& euler_abg)
{
    return euler_zyx_to_rotation(euler_abg.x(), euler_abg.y(), euler_abg.z());
}

Eigen::Matrix3d rotation_derivative_alpha(double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const double cg = std::cos(gamma), sg = std::sin(gamma);

    // The product rule gives dR/dalpha = Rz(gamma) Ry(beta) (dRx/dalpha), so the first column is 0
    // (Rx rotates about x and therefore leaves the x component alone).
    Eigen::Matrix3d derivative;
    derivative << 0.0, ca * sb * cg + sa * sg, -sa * sb * cg + ca * sg, 0.0, ca * sb * sg - sa * cg, -sa * sb * sg - ca * cg, 0.0, ca * cb, -sa * cb;
    return derivative;
}

Eigen::Matrix3d rotation_derivative_beta(double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const double cg = std::cos(gamma), sg = std::sin(gamma);

    // dR/dbeta = Rz(gamma) (dRy/dbeta) Rx(alpha).
    Eigen::Matrix3d derivative;
    derivative << -sb * cg, sa * cb * cg, ca * cb * cg, -sb * sg, sa * cb * sg, ca * cb * sg, -cb, -sa * sb, -ca * sb;
    return derivative;
}

Eigen::Matrix3d rotation_derivative_gamma(double alpha, double beta, double gamma)
{
    const double ca = std::cos(alpha), sa = std::sin(alpha);
    const double cb = std::cos(beta), sb = std::sin(beta);
    const double cg = std::cos(gamma), sg = std::sin(gamma);

    // dR/dgamma = (dRz/dgamma) Ry(beta) Rx(alpha) = [e3]x R, so row 1 is -row 2 of R, row 2 is row 1
    // of R, and row 3 is 0. That is why dr/dgamma simplifies to -n_y.x*v + n_y.y*u.
    Eigen::Matrix3d derivative;
    derivative << -cb * sg, -sa * sb * sg - ca * cg, -ca * sb * sg + sa * cg, cb * cg, sa * sb * cg - ca * sg, ca * sb * cg + sa * sg, 0.0, 0.0, 0.0;
    return derivative;
}

Eigen::Vector3d rotation_to_euler_zyx(const Eigen::Matrix3d& rotation)
{
    // Read the angles back out of the entries of R:
    //   R(2,0) = -sb, R(2,1) = sa*cb, R(2,2) = ca*cb, R(0,0) = cb*cg, R(1,0) = cb*sg
    // The asin argument is clamped, since numerical error can push its magnitude above 1.
    const double beta = std::asin(std::max(-1.0, std::min(1.0, -rotation(2, 0))));
    const double alpha = std::atan2(rotation(2, 1), rotation(2, 2));
    const double gamma = std::atan2(rotation(1, 0), rotation(0, 0));
    return Eigen::Vector3d(alpha, beta, gamma);
}

}  // namespace p2p_icp
