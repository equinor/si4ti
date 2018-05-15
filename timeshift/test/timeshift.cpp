#include <catch/catch.hpp>
#include <Eigen/Core>

#define TEST
#include <timeshift.cpp>

#include "matrices.hpp"

TEST_CASE("Regression test") {

    SECTION("BSpline") {

        SECTION("nt=101, nd = 0.1, ord = 3") {
            const auto expected = bspline_nt101_nd01_ord3();

            const auto knots = knotvector( 101, 0.1 );
            const auto result = bspline_matrix( 101, knots.data(), knots.size(), 3 );

            CHECK( expected.rows() == result.rows() );
            CHECK( expected.cols() == result.cols() );

            CHECK( result.isApprox(expected) );
        }

        SECTION("nt=20, nd = 0.3, ord = 5") {
            const auto expected = bspline_nt20_nd03_ord5();

            const auto knots = knotvector( 20, 0.3 );
            const auto result = bspline_matrix( 20, knots.data(), knots.size(), 5 );

            CHECK( expected.rows() == result.rows() );
            CHECK( expected.cols() == result.cols() );

            CHECK( result.isApprox(expected) );
        }

        SECTION("Normalized nt=20, nd = 0.3, ord = 5") {
            const auto expected = normalized_bspline();
            const auto result = normalized_bspline( 20, 0.3, 5 );

            CHECK( expected.rows() == result.rows() );
            CHECK( expected.cols() == result.cols() );

            CHECK( result.isApprox(expected, 1e-6) );
        }
    }

    SECTION("Constraints") {

        const auto expected = constraints();
        const auto spline = normalized_bspline( 101, 0.1, 3);
        const auto result = constraints( spline, 0.01, 0.03 );

        CHECK( expected.rows() == result.rows() );
        CHECK( expected.cols() == result.cols() );

        CHECK( result.isApprox(expected, 1e-5) );
    }

    auto vint1 = vintage1();
    const auto omega = angular_frequency( 30, 1.0 );
    const auto derived1 = derive( vint1, omega );

    SECTION("Derive") {
        const auto expected = derived();
        const auto result = derived1;

        CHECK( result.isApprox(expected, 1e-5) );
    }

    const auto spline = normalized_bspline( 30, 0.5, 3 );

    SECTION("Linear operator") {
        const auto expected = linearoperator();
        const auto result = linearoperator(derived1, spline);

        CHECK( expected.rows() == result.rows() );
        CHECK( expected.cols() == result.cols() );

        CHECK( result.isApprox(expected, 1e-5) );
    }

    SECTION("Solution") {
        auto vint2 = vintage2();
        const Eigen::Matrix<double, -1, 1> delta = vint2 - vint1;
        const auto derived2 = derive( vint2, omega );
        vector<double> derived = ( derived1 + derived2 ) / 2;

        const auto expected = solution();
        const auto result = solution( derived, delta, spline );

        CHECK( result.isApprox(expected, 1e-6) );
    }
}
