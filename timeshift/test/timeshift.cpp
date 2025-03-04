#include <catch/catch.hpp>
#include <Eigen/Core>

#include <segyio/segyio.hpp>

using namespace segyio::literals;

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
        expected << 0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5;
        Eigen::VectorXd corr(10);
        corr << 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5;
        apply_timeshift( x, corr );
        CHECK( x.isApprox( expected ) );

        expected << 0,1,2,3,4,5,6,7,8,9;
        apply_timeshift( x, (-corr).eval() );
        CHECK( x.isApprox( expected ) );
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

    std::vector< input_file > vintages;
    std::vector< std::string > fnames {
        "test-data/vintage1.sgy",
        "test-data/vintage2.sgy",
        "test-data/vintage3.sgy"
    };
    for( const auto& fname : fnames)
        vintages.emplace_back( segyio::path{ fname },
                               segyio::config{}
                                   .with( 5_il )
                                   .with( 21_xl ) );

    SECTION("Compute normalization") {
        auto f = normalize_surveys( float {30.0}, vintages );
        auto d = normalize_surveys( double{30.0}, vintages );
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
        CHECK( linsys.b.isApprox( -expectedb, 1e-5 ) );
    }
}

TEST_CASE("Linear algebra") {

    SECTION("4 traces, 3x3 per trace") {
        Eigen::MatrixXd vp_L( 12, 12 ); vp_L <<
          0.1, 0.4,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0.4, 0.2, 0.5,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0, 0.5, 0.3,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0, 0.5, 0.2,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0, 0.2, 0.4, 0.1,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0, 0.1, 0.3,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 0.1, 0.1,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0, 0.1, 0.1, 0.1,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0, 0.1, 0.1,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0, 0.2, 0.1,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0, 0.1, 0.2, 0.1,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 0.1, 0.2;

        Eigen::MatrixXd vp_L_c( 12, 2 ); vp_L_c <<
          0.1, 0.4,
          0.2, 0.5,
          0.3,   0,
          0.5, 0.2,
          0.4, 0.1,
          0.3,   0,
          0.1, 0.1,
          0.1, 0.1,
          0.1,   0,
          0.2, 0.1,
          0.2, 0.1,
          0.2,   0;

        Eigen::MatrixXd Bnn( 3, 3 ); Bnn <<
          1,4,0,
          4,2,5,
          0,5,3;

        SECTION("2 vintages, no horizontal smoothing") {
            Eigen::VectorXd x( 12 );
            x << 1,2,3,4,5,6,7,8,9,10,11,12;

            Eigen::MatrixXi comb( 1, 1 ); comb.setZero();

            const auto expected = vp_L * x;

            SECTION("2 inlines, 2 crosslines") {
                BlockBandedMatrix< double > bbm( vp_L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 2,
                                                 2, 2 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            SECTION("4 inlines, 1 crosslines") {
                BlockBandedMatrix< double > bbm( vp_L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 2,
                                                 4, 1 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            SECTION("1 inlines, 4 crosslines") {
                BlockBandedMatrix< double > bbm( vp_L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 2,
                                                 1, 4 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }
        }

        Eigen::VectorXd x( 24 );
        x << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24;

        SECTION("3 vintages, no horizontal smoothing") {
            Eigen::MatrixXi comb( 2, 2 ); comb.setZero();

            Eigen::MatrixXd L( 24, 24 );
            L.topLeftCorner( 12, 12 ) = vp_L;
            L.bottomLeftCorner( 12, 12 ) = 2 * vp_L;
            L.topRightCorner( 12, 12 ) = 3 * vp_L;
            L.bottomRightCorner( 12, 12 ) = 4 * vp_L;

            Eigen::MatrixXd L_c( 24, 4 );
            L_c.topLeftCorner( 12, 2 ) = vp_L_c;
            L_c.bottomLeftCorner( 12, 2 ) = 2 * vp_L_c;
            L_c.topRightCorner( 12, 2 ) = 3 * vp_L_c;
            L_c.bottomRightCorner( 12, 2 ) = 4 * vp_L_c;

            const auto expected = L * x;

            SECTION("2 inlines, 2 crosslines") {
                BlockBandedMatrix< double > bbm( L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 3,
                                                 2, 2 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            SECTION("4 inlines, 1 crosslines") {
                BlockBandedMatrix< double > bbm( L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 3,
                                                 4, 1 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            SECTION("1 inlines, 4 crosslines") {
                BlockBandedMatrix< double > bbm( L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 3,
                                                 1, 4 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }
        }

        SECTION("3 vintages, with horizontal smoothing") {
            Eigen::MatrixXi comb( 2, 2 );
            comb << 1,2,3,4;

            Eigen::MatrixXd smoothing( 12,12 );
            smoothing.setZero();

            Eigen::MatrixXd L_c( 24, 4 );
            L_c.topLeftCorner( 12, 2 ) = vp_L_c;
            L_c.bottomLeftCorner( 12, 2 ) = 2 * vp_L_c;
            L_c.topRightCorner( 12, 2 ) = 3 * vp_L_c;
            L_c.bottomRightCorner( 12, 2 ) = 4 * vp_L_c;

            SECTION("2 inlines, 2 crosslines") {
                smoothing.block( 0, 0, 3, 3 ) = 2 * Bnn;
                smoothing.block( 0, 3, 3, 3 ) = Bnn;
                smoothing.block( 0, 6, 3, 3 ) = Bnn;

                smoothing.block( 3, 3, 3, 3 ) = 2 * Bnn;
                smoothing.block( 3, 0, 3, 3 ) = Bnn;
                smoothing.block( 3, 9, 3, 3 ) = Bnn;

                smoothing.block( 6, 6, 3, 3 ) = 2 * Bnn;
                smoothing.block( 6, 9, 3, 3 ) = Bnn;
                smoothing.block( 6, 0, 3, 3 ) = Bnn;

                smoothing.block( 9, 9, 3, 3 ) = 2 * Bnn;
                smoothing.block( 9, 6, 3, 3 ) = Bnn;
                smoothing.block( 9, 3, 3, 3 ) = Bnn;

                Eigen::MatrixXd L( 24, 24 );
                L.topLeftCorner( 12, 12 ) = vp_L - comb(0,0)*smoothing;
                L.bottomLeftCorner( 12, 12 ) = 2*vp_L - comb(1,0)*smoothing;
                L.topRightCorner( 12, 12 ) = 3*vp_L - comb(0,1)*smoothing;
                L.bottomRightCorner( 12, 12 ) = 4*vp_L - comb(1,1)*smoothing;

                BlockBandedMatrix< double > bbm( L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 3,
                                                 2, 2 );
                const auto expected = L * x;
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            smoothing.block( 0, 0, 3, 3 ) = 3 * Bnn;
            smoothing.block( 0, 3, 3, 3 ) = Bnn;

            smoothing.block( 3, 3, 3, 3 ) = 2 * Bnn;
            smoothing.block( 3, 0, 3, 3 ) = Bnn;
            smoothing.block( 3, 6, 3, 3 ) = Bnn;

            smoothing.block( 6, 6, 3, 3 ) = 2 * Bnn;
            smoothing.block( 6, 9, 3, 3 ) = Bnn;
            smoothing.block( 6, 3, 3, 3 ) = Bnn;

            smoothing.block( 9, 9, 3, 3 ) = 3 * Bnn;
            smoothing.block( 9, 6, 3, 3 ) = Bnn;

            Eigen::MatrixXd L( 24, 24 );
            L.topLeftCorner( 12, 12 ) = vp_L - comb(0,0) * smoothing;
            L.bottomLeftCorner( 12, 12 ) = 2 * vp_L - comb(1,0) * smoothing;
            L.topRightCorner( 12, 12 ) = 3 * vp_L - comb(0,1) * smoothing;
            L.bottomRightCorner( 12, 12 ) = 4 * vp_L - comb(1,1) * smoothing;

            const auto expected = L * x;

            SECTION("1 inlines, 4 crossline") {
                BlockBandedMatrix< double > bbm( L_c,
                                                2,
                                                Bnn.sparseView(),
                                                comb,
                                                3,
                                                1, 4 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }

            SECTION("4 inlines, 1 crossline") {
                BlockBandedMatrix< double > bbm( L_c,
                                                 2,
                                                 Bnn.sparseView(),
                                                 comb,
                                                 3,
                                                 4, 1 );
                const auto result = bbm * x;
                CHECK( result.isApprox( expected ) );
            }
        }
    }
}

SCENARIO("Preconditioning") {

    GIVEN("A diagonal matrix M") {

        Eigen::MatrixXd vp_M( 10, 10 ); vp_M <<
            0.1,   0,   0,   0,   0,   0,   0,   0,   0,  0,
              0, 0.2,   0,   0,   0,   0,   0,   0,   0,  0,
              0,   0, 0.3,   0,   0,   0,   0,   0,   0,  0,
              0,   0,   0, 0.4,   0,   0,   0,   0,   0,  0,
              0,   0,   0,   0, 0.5,   0,   0,   0,   0,  0,
              0,   0,   0,   0,   0, 0.6,   0,   0,   0,  0,
              0,   0,   0,   0,   0,   0, 0.7,   0,   0,  0,
              0,   0,   0,   0,   0,   0,   0, 0.8,   0,  0,
              0,   0,   0,   0,   0,   0,   0,   0, 0.9,  0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,  1;

        // The second column should not be used by the preconditioner and is set
        // to 1 discover potential accidental off diagonal contributions
        Eigen::MatrixXd vp_M_c( 10, 2 ); vp_M_c <<
            0.1, 1,
            0.2, 1,
            0.3, 1,
            0.4, 1,
            0.5, 1,
            0.6, 1,
            0.7, 1,
            0.8, 1,
            0.9, 1,
            1, 1;

        // Bnn has no effect on the preconditioner but is required to
        // construct the compressed matrix
        Eigen::MatrixXd Bnn( 1, 1); Bnn.setOnes();


        WHEN("Preconditioning the system M * x") {
        THEN("2 vintage preconditioner should compute excact result") {

            Eigen::VectorXd x( 10 );
            x << 1,2,3,4,5,6,7,8,9,10;

            Eigen::VectorXd b = vp_M * x;

            Eigen::MatrixXi comb( 1, 1 );
            BlockBandedMatrix< double > bbm( vp_M_c,
                                             2,
                                             Bnn.sparseView(),
                                             comb,
                                             2,
                                             2, 5 );

            Si4tiPreconditioner< double > pcon;
            pcon.initialize( bbm );
            const auto result = pcon.solve( b );
            CHECK( result.isApprox( x ) );
        }}

        WHEN("Preconditioning the system [ M, 0; 0, 2*M ] * x") {
        THEN("3 vintage preconditioner should compute excact result") {
            Eigen::VectorXd x( 20 );
            x << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20;

            Eigen::MatrixXd M( 20, 20 ); M.setZero();
            M.topLeftCorner( 10, 10 ) = vp_M;
            M.bottomRightCorner( 10, 10 ) = 4*vp_M;

            Eigen::VectorXd b = M * x;

            Eigen::MatrixXd M_c( 20, 4 ); M_c.setZero();
            M_c.topLeftCorner( 10, 2 ) = vp_M_c;
            M_c.bottomRightCorner( 10, 2 ) = 4*vp_M_c;

            Eigen::MatrixXi comb( 2, 2 ); comb.setZero();
            BlockBandedMatrix< double > bbm( M_c,
                                             2,
                                             Bnn.sparseView(),
                                             comb,
                                             3,
                                             2, 5 );

            Si4tiPreconditioner< double > pcon;
            pcon.initialize( bbm );
            const auto result = pcon.solve( b );
            CHECK( result.isApprox( x ) );
        }}

    }

}

SCENARIO( "Infer sampling interval ") {
    double sampling_interval;

    input_file f( segyio::path{ "test-data/vintage1.sgy" },
                     segyio::config{}.with( segyio::ilbyte{ 5 } )
                                     .with( segyio::xlbyte{ 21 } ) );

    GIVEN( "the sampling interval is user-spesified" ) {
        sampling_interval = infer_interval( f, 5.0 );
        CHECK(sampling_interval == 5.0 );
    }

    GIVEN( "the sampling interval is not user spesified" ) {
        sampling_interval = infer_interval( f, 0.0 );;
        CHECK(sampling_interval == 4.0 );
    }
}
