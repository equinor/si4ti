#include <catch/catch.hpp>
#include <Eigen/Core>

#include <segyio/segyio.hpp>

#include <impedance.hpp>

//int Progress::count = 0;
//int Progress::expected = 100000; // Set high to avoid progress printouts

using T = double;

TEST_CASE( "Conjugate gradient" ) {

    SECTION( "Dense matrix" ) {
        matrix< T > AA( 50, 50 );
        AA.setRandom();

        matrix< T > A = AA.transpose() * AA;

        vector< T > x( 50 );
        x.setRandom();

        vector< T > expected = x;
        vector< T > b = A * x;
        x.setRandom();

        x = conjugate_gradient( A, x, b, 100 );

        CHECK( x.isApprox( expected, 1e-10 ) );
    }

    SECTION( "Si4ti matrix" ) {
        std::vector< matrix< T > > m{ matrix< T >( 2, 2 ),
                                      matrix< T >( 2, 2 ) };
        m[0] << 10, 11, 11, 13; m[1] << 20, 21, 21, 23;

        vector< T > x( 64 );
        x.setRandom();

        Si4tiImpMatrix< T > rbdm( m, 2, 4, 4, 0.05, 0.0001, 4.0, false );

        vector< T > b = rbdm * x;
        vector< T > expected = x;
        x.setRandom();

        x = conjugate_gradient( rbdm, x, b, 50 );

        CHECK( x.isApprox( expected, 1e-10 ) );
    }
}
