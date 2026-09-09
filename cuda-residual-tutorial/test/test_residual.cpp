#include "cuda_residual_tutorial/residual.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tutorial = cuda_residual_tutorial;

void expect_near(double actual, double expected)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12)
    {
        throw std::runtime_error("incorrect residual: expected " + std::to_string(expected) + ", got " + std::to_string(actual));
    }
}

int main()
{
    try
    {
        const tutorial::Pose identity;
        if (!tutorial::evaluate_cuda({}, identity).empty())
        {
            throw std::runtime_error("empty batch should return no residuals");
        }
        // Exact 90-degree Z rotation catches row/column-major and transform-sign errors.
        const tutorial::Pose pose{{0, -1, 0, 1, 0, 0, 0, 0, 1}, {0.5, -1, 2}};
        const tutorial::Pair pair{{1, 2, 3}, {-2, 1, 4}, {0, -1, 0}};
        // R*x+t = (-1.5, 0, 5), so r = (0.5, -1, 1), n dot r = 1.
        // Include partial blocks and a second block with varying data/order.
        for (const std::size_t count : {1u, 255u, 256u, 257u, 513u})
        {
            std::vector<tutorial::Pair> pairs(count, pair);
            for (std::size_t i = 0; i < count; ++i)
            {
                pairs[i].target[2] += static_cast<double>(i);
            }
            const auto results = tutorial::evaluate_cuda(pairs, pose);
            if (results.size() != count)
            {
                throw std::runtime_error("output count differs from input count");
            }
            for (std::size_t i = 0; i < count; ++i)
            {
                expect_near(results[i].point[0], 0.5);
                expect_near(results[i].point[1], -1);
                expect_near(results[i].point[2], 1 - static_cast<double>(i));
                expect_near(results[i].plane, 1);
            }
        }
        // Signed plane residual with a normal that uses all three coordinates.
        const auto result = tutorial::evaluate_cuda({{{1, 2, 3}, {2, 4, 5}, {1.0 / 3, 2.0 / 3, 2.0 / 3}}}, identity);
        expect_near(result[0].plane, -3);
        std::cout << "Known residuals, empty input, batch boundaries, and output order passed.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
