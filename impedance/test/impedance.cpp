#include <catch/catch.hpp>
#include <Eigen/Core>

#include <segyio/segyio.hpp>

#include <impedance.hpp>
#include "matrixes.hpp"

int Progress::count = 0;
int Progress::expected = 100000; // Set high to avoid progress printouts

using T = double;

TEST_CASE("Regression test") {

    SECTION( "segmentation 1 segment" ) {
        std::vector< std::pair< int, int > > expected{{0,65708}};
        auto result = segments( 1, 149, 441, 50 );

        CHECK( result == expected );
    }

    SECTION( "segmentation 2 segments" ) {
        std::vector< std::pair< int, int > > expected{{0,55565},{33075,65708}};
        auto result = segments( 2, 149, 441, 50 );

        CHECK( result == expected );
    }

    SECTION("myhamn") {
        vector< T > expected( 10 ); expected <<
            0.00000000000000e+000,
            116.977778440511e-003,
            413.175911166535e-003,
            750.000000000000e-003,
            969.846310392954e-003,
            969.846310392954e-003,
            750.000000000000e-003,
            413.175911166535e-003,
            116.977778440511e-003,
            0.00000000000000e+000
        ;

        vector< T > result = myhamn< T >( 10 );
        CHECK( result.isApprox( expected ) );
    }

    SECTION("mask") {
        matrix< T > expected( 5, 5 ); expected <<
            848.283639957513e-003, 788.262245442287e-003, 611.472388571261e-003, 373.470531082249e-003, 180.924341549619e-003,
            788.262245442287e-003, 848.283639957513e-003, 788.262245442287e-003, 611.472388571261e-003, 373.470531082249e-003,
            611.472388571261e-003, 788.262245442287e-003, 848.283639957513e-003, 788.262245442287e-003, 611.472388571261e-003,
            373.470531082249e-003, 611.472388571261e-003, 788.262245442287e-003, 848.283639957513e-003, 788.262245442287e-003,
            180.924341549619e-003, 373.470531082249e-003, 611.472388571261e-003, 788.262245442287e-003, 848.283639957513e-003
        ;

        matrix< T > result = mask< T >(5);
        CHECK( result.isApprox( expected ) );
    }

    std::vector< std::string > fnames {
        "test-data/vintage1.sgy",
        "test-data/vintage2.sgy",
        "test-data/vintage3.sgy"
    };
    std::vector< input_file > vintages;
    for( const auto& fname : fnames)
        vintages.push_back( { segyio::path{ fname },
                              segyio::config{}.with( 5_il )
                                              .with( 21_xl ) } );

    SECTION("time varying wavelet") {
        auto expected = tvar_wvlt();
        auto result = timevarying_wavelet< T >( vintages[0] );

        CHECK( result.isApprox( expected, 1e-5 ) );
    }

    SECTION("time invariant wavelet") {
        auto expected = tinvar_wvlt();
        auto result = timeinvariant_wavelet< T >( vintages[0] );

        CHECK( result.isApprox( expected, 1e-5 ) );
    }

    std::vector< matrix< T > > tvr_wvlts = wavelets< T >( vintages, true, 1 );
    T norm_tvw = normalization( tvr_wvlts );
    auto fwd_op_tvw = forward_operators< T >( tvr_wvlts,
                                              vintages.size(),
                                              norm_tvw );

    SECTION("forward operator (time varying wavelet)") {
        auto expected = frwd_op_tvw();

        for( int i = 0; i < expected.size(); ++i ) {
            CHECK( fwd_op_tvw[i].isApprox( expected[i], 2e-5 ) );
        }
    }

    std::vector< matrix< T > > tinvr_wvlts = wavelets< T >( vintages, false, 1 );
    T norm_tinw = normalization( tinvr_wvlts );
    auto fwd_op_tinw = forward_operators< T >( tinvr_wvlts,
                                               vintages.size(),
                                               norm_tinw );

    SECTION("forward operator (time invariant wavelet)") {
        auto expected = frwd_op_tinw();

        for( int i = 0; i < expected.size(); ++i ) {
            CHECK( fwd_op_tinw[i].isApprox( expected[i], 3e-5 ) );
        }
    }

    const T damping = 0.0001;
    auto in_op_tvw = inverse_operators< T >( fwd_op_tvw, 3, damping );

    SECTION("inverse operator (time varying wavelet)") {
        auto expected = inv_op_tvw();

        for( int i = 0; i < expected.size(); ++i ) {
            CHECK( in_op_tvw[i].isApprox( expected[i], 2e-5 ) );
        }
    }

    auto in_op_tinw = inverse_operators< T >( fwd_op_tinw, 3, damping );

    SECTION("inverse operator (time invariant wavelet)") {
        auto expected = inv_op_tinw();

        for( int i = 0; i < expected.size(); ++i ) {
            CHECK( in_op_tinw[i].isApprox( expected[i], 1e-5 ) );
        }
    }

    SECTION("1D solution (time varying wavelet)") {
        auto expected = sol_1D_tvw();
        auto result = solve_1D< T >( vintages,
                                     in_op_tvw,
                                     fwd_op_tvw,
                                     damping,
                                     0,
                                     329);

        CHECK( result.b.isApprox( expected, 2e-5 ) );
    }

    SECTION("1D solution (time invariant wavelet)") {
        auto expected = sol_1D_tinw();
        auto result = solve_1D< T >( vintages,
                                     in_op_tinw,
                                     fwd_op_tinw,
                                     damping,
                                     0,
                                     329);

        CHECK( result.b.isApprox( expected, 2e-5 ) );
    }

    SECTION("1D solution segmented (time varying wavelet)") {
        auto expected = sol_seg_1D_tvw();
        auto result = solve_1D< T >( vintages,
                                     in_op_tvw,
                                     fwd_op_tvw,
                                     damping,
                                     180,
                                     329);

        CHECK( result.b.isApprox( expected, 2e-5 ) );
    }

    SECTION("Matrix") {
        Eigen::VectorXd expected( 64 );
        expected <<
            1.04999947547913, 1.25000000000000, 1.07099962234497,
            1.27500009536743, 1.09200000762939, 1.30000007152557,
            1.11299967765808, 1.32499957084656, 1.25999987125397,
            1.49999988079071, 1.28099954128265, 1.52500057220459,
            1.30200004577637, 1.55000007152557, 1.32300019264221,
            1.57500004768372, 1.46999990940094, 1.75000035762787,
            1.49100005626678, 1.77500009536743, 1.51200056076050,
            1.80000007152557, 1.53300023078918, 1.82499969005585,
            1.68000030517578, 2.00000047683716, 1.70099997520447,
            2.02500009536743, 1.72200059890747, 2.04999995231628,
            1.74300062656403, 2.07500004768372, 43.1500015258789,
            47.3499984741211, 43.1909980773926, 47.3950004577637,
            43.2319984436035, 47.4399986267090, 43.2730026245117,
            47.4850006103516, 43.5600013732910, 47.7999992370605,
            43.6009979248047, 47.8449974060059, 43.6419982910156,
            47.8899993896484, 43.6829986572266, 47.9349975585938,
            43.9699974060059, 48.2499961853027, 44.0110015869141,
            48.2950019836426, 44.0519981384277, 48.3399963378906,
            44.0929985046387, 48.3850021362305, 44.3799972534180,
            48.7000007629395, 44.4210014343262, 48.7450027465820,
            44.4619979858398, 48.7899971008301, 44.5029983520508,
            48.8349990844727;

        std::vector< Eigen::MatrixXd > m{ Eigen::MatrixXd( 2, 2 ),
                                          Eigen::MatrixXd( 2, 2 ) };
        m[0] << 10, 11, 12, 13; m[1] << 20, 21, 22, 23;

        Eigen::VectorXd x( 64 );
        x << 0.000, 0.100, 0.001, 0.101, 0.002, 0.102, 0.003, 0.103,
             0.010, 0.110, 0.011, 0.111, 0.012, 0.112, 0.013, 0.113,
             0.020, 0.120, 0.021, 0.121, 0.022, 0.122, 0.023, 0.123,
             0.030, 0.130, 0.031, 0.131, 0.032, 0.132, 0.033, 0.133,
             1.000, 1.100, 1.001, 1.101, 1.002, 1.102, 1.003, 1.103,
             1.010, 1.110, 1.011, 1.111, 1.012, 1.112, 1.013, 1.113,
             1.020, 1.120, 1.021, 1.121, 1.022, 1.122, 1.023, 1.123,
             1.030, 1.130, 1.031, 1.131, 1.032, 1.132, 1.033, 1.133;

        SimpliImpMatrix< T > rbdm( m, 2, 4, 4, 0.05, 0.0001, 4.0, false );

        auto result = rbdm * x;

        CHECK( result.isApprox( expected, 1e-5 ) );
    }

    std::vector< std::string > rel_AI_fnames {
        "test-data/relAI-0.sgy",
        "test-data/relAI-1.sgy",
        "test-data/relAI-2.sgy"
    };
    std::vector< output_file > rel_AI_files;
    for( const auto& fname : rel_AI_fnames )
        rel_AI_files.push_back( { segyio::path{ fname },
                                  segyio::config{}.with( 5_il )
                                                  .with( 21_xl )} );

    SECTION("solution (time varying wavelet)") {
        auto expected = sol_tvw();

        auto result = compute_impedance< T >( vintages,
                                              rel_AI_files,
                                              fwd_op_tvw,
                                              norm_tvw,
                                              50,
                                              0.0001, 0.0001, 0.05, 4,
                                              0, 329 );

        CHECK( result.isApprox( expected, 8e-5 ) );
    }

    SECTION("solution segmented (time varying wavelet)") {
        auto expected = sol_seg_tvw();

        auto result = compute_impedance< T >( vintages,
                                              rel_AI_files,
                                              fwd_op_tvw,
                                              norm_tvw,
                                              50,
                                              0.0001, 0.0001, 0.05, 4,
                                              180, 329 );

        CHECK( result.isApprox( expected, 7e-4 ) );
    }

}
