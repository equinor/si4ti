#include <catch/catch.hpp>
#include <Eigen/Core>

#include <segyio/segyio.hpp>

#include <timeshift.hpp>

#include "matrices.hpp"
#include "matchers.hpp"

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

    SECTION("Shift data") {
        Eigen::VectorXd x(10);
        x << 0,1,2,3,4,5,6,7,8,9;
        Eigen::VectorXd expected(10);
        expected << -0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5;
        Eigen::VectorXd corr(10);
        corr << 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5;
        apply_timeshift( x, corr );
        CHECK_THAT( expected.transpose().eval(), ApproxRange(x.transpose().eval()) );
    }

    SECTION("Mask linear operator") {
        Eigen::MatrixXi expected( 3, 3 );

        expected << 1,1,1,
                    1,1,1,
                    1,1,1;
        auto result = mask_linear( 4, 0, 3 );
        CHECK( result == expected );

        expected << 0,0,0,
                    0,1,1,
                    0,1,1;
        result = mask_linear( 4, 1, 3 );
        CHECK( result == expected );

        expected << 0,0,0,
                    0,1,0,
                    0,0,0;
        result = mask_linear( 4, 1, 2 );
        CHECK( result == expected );
    }

    SECTION("Mask solution") {
        Eigen::VectorXi expected( 3 );

        expected << 1,1,1;
        auto result = mask_solution( 4, 0, 3 );
        CHECK( result == expected );

        expected << 0,1,1;
        result = mask_solution( 4, 1, 3 );
        CHECK( result == expected );

        expected << 0,1,0;
        result = mask_solution( 4, 1, 2 );
        CHECK( result == expected );
    }

    SECTION("Multiplier") {
        Eigen::MatrixXi expected( 3, 3 );

        expected << 3,2,1,
                    2,4,2,
                    1,2,3;
        auto result = multiplier( 4 );
        CHECK( result == expected );
    }
}

TEST_CASE("3 vintages (tiny cubes)") {

    std::vector< sio::simple_file > vintages;
    std::vector< std::string > fnames {
        "test-data/vintage1.sgy",
        "test-data/vintage2.sgy",
        "test-data/vintage3.sgy"
    };
    for( const auto& fname : fnames)
        vintages.push_back( { fname, sio::config().ilbyte(  5 )
                                                  .xlbyte( 21 )} );

    SECTION("Compute normalization") {
        auto f = normalize_surveys( 30.0f, vintages );
        auto d = normalize_surveys( 30.0d, vintages );
        CHECK( f == Approx( 2.0613483677 ) );
        CHECK( d == Approx( 2.0613483677 ) );
    }

    SECTION("Linear system") {
        const geometry geo{ 51, 12, 3, 4 };
        const int splineord = 3;
        const auto spline = normalized_bspline( geo.samples, 0.1, splineord );
        const auto C = constraints( spline, 0.01, 0.03 );
        const auto omega = angular_frequency( geo.samples, 1.0 );
        const int ndiagonals = splineord - 1;
        const double normalizer = 3;

        const auto linsys = build_system( spline,
                                          C,
                                          omega,
                                          normalizer,
                                          vintages,
                                          geo,
                                          ndiagonals );

        const auto expectedL = linsysL();
        const auto expectedb = linsysb();
        CHECK( linsys.L.isApprox( expectedL, 1e-5 ) );
        CHECK( linsys.b.isApprox( expectedb, 1e-5 ) );
    }
}
