#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <getopt.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>

#include <segyio/segy.h>

namespace {

struct options {
    std::vector< std::string > files;
    double timeshift_resolution = 0.05;
    double horizontal_smoothing = 0.01;
    double vertical_smoothing   = 0.1;
    int    solver_max_iter      = 100;
    bool   double_precision     = false;
    bool   correct_4d_noise     = false;
    double normalizer           = 30;
    int    verbosity            = 0;
    int    ilbyte               = SEGY_TR_INLINE;
    int    xlbyte               = SEGY_TR_CROSSLINE;
};

options parseopts( int argc, char** argv ) {
    static struct option longopts[] = {
        { "timeshift-resolution", required_argument, 0, 'r' },
        { "horizontal-smoothing", required_argument, 0, 'H' },
        { "vertical-smoothing",   required_argument, 0, 'V' },
        { "max-iter",             required_argument, 0, 'm' },
        { "double-precision",     no_argument,       0, 'd' },
        { "correct-4D",           no_argument,       0, 'c' },
        { "normalizer",           required_argument, 0, 'n' },
        { "ilbyte",               required_argument, 0, 'i' },
        { "xlbyte",               required_argument, 0, 'x' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    options opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "r:H:V:m:dcn:i:x:v",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'r': break;
            case 'H': break;
            case 'V': break;
            case 'm': break;
            case 'd': break;
            case 'c': break;
            case 'n': break;
            case 'i': opts.ilbyte = std::stoi( optarg ); break;
            case 'x': opts.xlbyte = std::stoi( optarg ); break;
            case 'v': break;
            case 'h': break;

            default:
                std::exit( 1 );
        }
    }

    if( argc - optind < 2 ) {
        std::cerr << "Need at least 2 input files\n";
        std::exit( 1 );
    }

    while( optind < argc )
        opts.files.push_back( argv[ optind++ ] );

    return opts;
}

/*
 * Analyse the B-spline with De Boor's algorithm as an n * basis-functions
 * matrix.
 *
 * The samples 't' is the linear space [0,1] of interval dx = 1/(n-1), padded on
 * both sides according to the order of the spline curve, resulting in the
 * following intervals in start/step/stop notation:
 *
 * | t < 0:       [-dx * padlen : dx : 0              )
 * | t <= 0 <= 1: [0            : dx : 1              ]
 * | t > 1:       [1 + dx       : dx : padlen * dx + 1]
 *
 * The knots knotv are padded similarly, depending on the order of the curve,
 * so that:
 * | pre:   [-dx * order : dx : -dx * (order - padlen)]
 * | knots: knotv
 * | post:  [1 + dx      : dx : 1 + padlen * dx       ]
 *
 * Please note the two padlens are different for samples and knots.
 */
template< typename Numeric, int... Options >
Eigen::Matrix< Numeric, Eigen::Dynamic, Eigen::Dynamic, Options... >
bspline_matrix( int n, const Numeric* knotv, int knotlen, int order ) {
    using array = Eigen::Array< Numeric, Eigen::Dynamic, 1 >;
    using matrix = decltype( bspline_matrix( n, knotv, knotlen, order ) );

    auto samples = []( int n, int order ) {
        const Numeric interval = 1.0 / (n - 1.0);
        const auto padlen = 10 * order;
        array t( 2 * padlen + n );

        for( int i = 0; i < padlen; ++i )
            t[i] = -interval * (padlen - i);

        for( int i = 0; i < n; ++i )
            t[padlen + i] = i * interval;

        for( int i = 1; i <= padlen; ++i )
            t[padlen + n + i-1] = 1.0 + i*interval;

        return t;
    };

    auto padknots = []( int n, const Numeric* knotv, int knotlen, int order ) {
        const Numeric interval = 10.0 / (n - 1.0);

        const auto padlen = order + 1;
        std::vector< Numeric > knots( knotlen + 2*padlen );

        for( int i = 0; i <= padlen; ++i )
            knots[i] = -interval * (order - i);

        const auto scale = std::accumulate(
            knotv,
            knotv + knotlen,
            false,
            []( bool acc, Numeric x ) { return acc or x <= 0 or 1 <= x; }
        );

        const auto denom = scale ? n + 1 : 1;

        std::transform( knotv,
                        knotv + knotlen,
                        knots.begin() + padlen,
                        [=]( Numeric x ) { return x / denom; }
        );

        for( int i = 0; i < padlen; ++i )
            knots[padlen + knotlen + i] = 1.0 + i*interval;

        return knots;
    };

    const auto t = samples( n, order );
    const auto knots = padknots( n, knotv, knotlen, order );

    matrix P = matrix::Zero( t.size(), knots.size() - 1 );

    for( int i = 0; i < int(knots.size()) - 1; ++i ) {
        const auto low = knots[ i ];
        const auto high = knots[ i + 1 ];
        P.col(i) = (low <= t && t < high).template cast< Numeric >();
    }

    array P1, P2;
    for( int j = 1; j <= order; ++j ) {
        for( int i = 0; i < int(knots.size()) - (j + 1); ++i ) {

            const auto low  = knots[i];
            const auto mid  = knots[i + j];
            const auto high = knots[i + j + 1];

            const auto& t1 = (t - low) / (mid - low);
            const auto& t2 = (high - t) / (high - knots[i+1]);

            P1 = P.col( i );
            P2 = P.col( i+1 );

            P.col( i ) = (t1 * P1) + (t2 * P2);
        }
    }

    const auto col = 0;
    const auto colsz = P.cols() - order;
    const auto row = 10 * order;
    const auto rowsz = n;

    return P.block( row, col, rowsz, colsz );
}

using filehandle = std::unique_ptr< segy_file, decltype( &segy_close ) >;

filehandle openfile( const std::string& fname, std::string mode = "rb" ) {
    if( mode.back() != 'b' ) mode.push_back( 'b' );

    filehandle fp( segy_open( fname.c_str(), mode.c_str() ), &segy_close );
    if( !fp ) throw std::invalid_argument( "Unable to open file " + fname );

    return fp;
}

struct geometry {
    int samples;
    int traces;
    int trace_bsize;
    long trace0;
    int ilines;
    int xlines;
};

geometry findgeometry( segy_file* fp, int ilbyte, int xlbyte ) {
    // TODO: proper error handling other than throw exceptions without checking
    // what went wrong
    char header[ SEGY_BINARY_HEADER_SIZE ] = {};

    auto err = segy_binheader( fp, header );
    if( err != SEGY_OK ) throw std::runtime_error( "Unable to read header" );

    geometry geo;
    geo.samples     = segy_samples( header );
    geo.trace0      = segy_trace0( header );
    geo.trace_bsize = segy_trace_bsize( geo.samples );

    err = segy_traces( fp, &geo.traces, geo.trace0, geo.trace_bsize );

    if( err != SEGY_OK )
        throw std::runtime_error( "Could not compute trace count" );

    err = segy_lines_count( fp,
                            ilbyte, xlbyte,
                            SEGY_INLINE_SORTING,
                            1, // offsets
                            &geo.ilines,
                            &geo.xlines,
                            geo.trace0,
                            geo.trace_bsize );

    if( err != SEGY_OK )
        throw std::runtime_error( "Could not count lines" );

    return geo;
}

template< typename Vector >
void writefile( segy_file* basefile,
                const Vector& v,
                const std::string& fname,
                const geometry& geo ) {

    char textheader[ SEGY_TEXT_HEADER_SIZE ];
    char binaryheader[ SEGY_BINARY_HEADER_SIZE ];
    char traceheader[ SEGY_TRACE_HEADER_SIZE ];
    int err = 0;

    auto fp = openfile( fname.c_str(), "w+b" );

    err = segy_read_textheader( basefile, textheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to read text header" );
    err = segy_write_textheader( fp.get(), 0, textheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to write text header" );

    err = segy_binheader( basefile, binaryheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to read binary header" );
    err = segy_write_binheader( fp.get(), binaryheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to write binary header" );

    auto buf = v.data();
    for( int traceno = 0; traceno < geo.traces; ++traceno ) {
        err = segy_traceheader( basefile,
                                traceno,
                                traceheader,
                                geo.trace0,
                                geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to read trace header "
                                    + std::string( traceno ) );

        err = segy_write_traceheader( fp.get(),
                                      traceno,
                                      traceheader,
                                      geo.trace0,
                                      geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to write trace header "
                                    + std::string( traceno ) );

        err = segy_writetrace( fp.get(),
                               traceno,
                               buf,
                               geo.trace0,
                               geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to write trace "
                                    + std::string( traceno ) );

        buf += geo.samples;
    }
}

template< typename T >
using vector = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
template< typename T >
using matrix = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;
template< typename T >
using sparse = Eigen::SparseMatrix< T >;

template< typename T >
std::vector< T > knotvector( int samples, T density ) {
    const auto step = T(1) / density;
    const auto middle = T(samples + 1) / 2;

    std::vector< T > knotv;
    for( auto f = middle; f > 1 / density; f -= step )
        knotv.push_back( f );

    std::reverse( knotv.begin(), knotv.end() );

    for( auto f = middle + step; f < (1+samples) - 1/density; f += step )
        knotv.push_back( f );

    return knotv;
}

template< typename T >
matrix< T > normalized_bspline( int samples, T density, int degree ) {
    const auto knotv = knotvector( samples, density );
    auto B = bspline_matrix( samples, knotv.data(), knotv.size(), degree );
    return B * B.colwise().sum().cwiseInverse().asDiagonal();
}

template< typename T >
matrix< T > constraints( const matrix< T >& spline,
                         T vertical_smoothing,
                         T horizontal_smoothing ) {

    /*
     * Smoothing constraints
     *
     * MxM matrix (M = number of spline functions) containing the vertical
     * smoothing and central component of the horizontal smoothing.
     *
     */

    auto basis_squared = spline.transpose() * spline;
    const auto tracelen = spline.rows();

    /*
     * The matrix D computes an array of differences d[n] - d[n+1] representing
     * level of vertical smoothness. It's assumed traces are vertically
     * consistent. D is used to prefer solutions with minimal difference
     * between consecutive timeshift values.
     *
     * It is a tracelen-1 x tracelen matrix on the form:
     *
     *               1 -1  0 .. 0  0
     *               0  1 -1 .. 0  0
     *        D  =   .  .  . .. .  .
     *               .  .  . .. .  .
     *               0  0  0 .. 1 -1
     */
    matrix< T > D = matrix< T >::Identity( tracelen - 1, tracelen );
    D.template diagonal< 1 >().fill( -1.0 );
    matrix< T > Dpm = spline.transpose() * D.transpose() * D * spline;

    return (horizontal_smoothing * basis_squared) + (vertical_smoothing * Dpm);
}

template< typename T >
vector< T > frequency_spectrum( int n, T dt = 1 ) {
    /*
     * Build the frequency spectrum for use in fft later
     *
     * f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
     * f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd
     *
     * Basically a straight-up implementation based on the numpy docs
     *
     * https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html
     */
    const auto even = n % 2 == 0;
    const auto half = (even ? n : (n-1)) / 2;

    vector< T > f( n );
    f.head( half + 1 ).setLinSpaced( half + 1, 0, half );
    f.tail( half + 0 ).setLinSpaced( half, -half, -1 );
    return f / (dt * n);
}

template< typename T >
vector< T > angular_frequency( int n, T dt = 1 ) {
    constexpr static const T pi = 3.14159265358979323846;
    return 2 * pi * frequency_spectrum( n, dt );
}

template< typename T >
vector< T >& derive( vector< T >& signal, const vector< T >& omega ) {

    /*
     * D = FFTI(iωFFT(signal))
     * where D is the derivative of a signal (data trace)
     *
     * It takes the entire omeg an argument, in order to not recompute it on
     * every invocation (it's shared across all derivations in this
     * application).
     */

    // TODO: MUST be initialised outside
    static Eigen::FFT< T > fft;
    vector< std::complex< T > > ff;

    fft.fwd( ff, signal );
    ff.array() *= std::complex< T >(0, 1) * omega.array();
    fft.inv( signal , ff );
    return signal;
}

template< typename T >
matrix< T > linearoperator( const vector< T >& derived,
                            const matrix< T >& spline ) {
    /*
     * Compute the linear operator
     *
     * Returns an MxM matrix, where M is number of spline functions
     */
    auto Lt = derived.asDiagonal() * spline;
    return Lt.transpose() * Lt;
}

template< typename T >
auto solution( const vector< T >& derived,
               const vector< T >& delta,
               const matrix< T >& spline )
    -> decltype( (derived.asDiagonal() * spline).transpose() * delta ) {
    /*
     * Solution (right-hand-side of the system)
     *
     * Returns an N dimensional column vector, where N is number of spline
     * functions
     *
     * TODO: remove allocation, do inline
     */
    return (derived.asDiagonal() * spline).transpose() * delta;
}


template< typename Vector >
Vector& gettr( segy_file* vintage,
               int traceno,
               const geometry& geo,
               Vector& buffer ) {

    int err = 0;
    std::vector< float > trbuf( geo.samples );

    err = segy_readtrace( vintage,
                          traceno,
                          trbuf.data(),
                          geo.trace0,
                          geo.trace_bsize );

    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to read trace "
                                + std::to_string( traceno ) );

    segy_to_native( SEGY_IBM_FLOAT_4_BYTE,
                    geo.samples,
                    trbuf.data() );

    std::copy( trbuf.begin(), trbuf.end(), buffer.data() );

    return buffer;
}

template< typename T >
std::vector< Eigen::Triplet< T > > getBnn( const matrix< T >& B,
                                           T horizontal_smoothing ) {

    sparse< T > Bnn = (B.transpose() * B).sparseView();

    decltype( getBnn( B, horizontal_smoothing ) ) triplets;
    for( int k = 0; k < Bnn.outerSize(); ++k ) {
        for( typename decltype( Bnn )::InnerIterator it( Bnn, k ); it; ++it ) {
            const auto row = it.row();
            const auto col = it.col();
            const auto val = it.value() * -.25 * horizontal_smoothing;
            triplets.emplace_back( row, col, val );
        }
    }

    return triplets;
}

template< typename T >
std::pair< sparse< T >, vector< T > >
L_ij( segy_file* vintage1,
                               segy_file* vintage2,
                               const geometry& geo,
                               const matrix< T >& B,
                               const matrix< T >& C,
                               const std::vector< Eigen::Triplet< T > >& Bnn,
                               const vector< T >& omega,
                               T normalizer ) {

    const auto dims = B.cols();

    int err = 0;

    /* reuse a bunch of vectors, as they're the same size throughout and
     * constructing them is expensive
     */
    vector< T > tr1( geo.samples );
    vector< T > tr2( geo.samples );
    vector< T > delta( geo.samples );
    vector< T > D( geo.samples );
    matrix< T > Lt( dims, dims );
    sparse< T > picoL( dims, dims );
    sparse< T > L( dims * geo.traces, dims * geo.traces );
    vector< T > b( dims * geo.traces );

    using inneritr = typename decltype( picoL )::InnerIterator;

    // TODO: evaluate triplets-then-insert vs. reserve-and-insert per trace and
    // reserve-and-insert per vintage pair (full-file)
    std::vector< Eigen::Triplet< T > > triplets;

    int j = 0, k = 0;
    for( auto traceno = 0; traceno < geo.traces; ++traceno ) {
        tr1 = gettr( vintage1, traceno, geo, tr1 ) / normalizer;
        tr2 = gettr( vintage2, traceno, geo, tr2 ) / normalizer;

        // derive works in-place and will overwrite its input argument
        // signal, so delta must be computed first
        delta = tr2.array() - tr1.array();

        D = 0.5 * (derive( tr1, omega ) + derive( tr2, omega ));
        Lt = linearoperator( D, B );
        picoL = (Lt + C).sparseView();

        for( int i = 0; i < picoL.outerSize(); ++i ) {
            for( inneritr it( picoL, i ); it; ++it ) {
                /*
                 * position each sample into the linear operator based on the
                 * two vintages, by simply shifting them along the diagonal of
                 * the larger system
                 */
                const auto row = traceno*dims + it.row();
                const auto col = traceno*dims + it.col();
                triplets.emplace_back( row, col, it.value() );
            }
        }

        b.segment( traceno * dims, dims ) = solution( D, delta, B );


        const auto i = traceno;
        // C(j-1, k)
        const auto i1 = k == 0 ? i : i - 1;
        // C(j+1, k)
        const auto i2 = k == geo.xlines - 1 ? i : i + 1;
        // C(j, k-1)
        const auto i3 = j == 0 ? i : i - geo.xlines;
        // C(j, k+1)
        const auto i4 = j == geo.ilines - 1 ? i : i + geo.ilines;

        // move one xline forward, unless we're at the last xline in this
        // inline, then wrap around to zero again
        k = k + 1 >= geo.xlines ? 0 : k + 1;
        // move the ilines forwards every time xlines wrap around
        if( k == 0 ) ++j;

        for( auto is : { i1, i2, i3, i4 } ) {
            for( const auto& it : Bnn ) {
                // TODO: consider extraction to emphasise that rows does not
                // move as much as cols
                const auto row = is*dims + it.row();
                const auto col = traceno*dims + it.col();
                triplets.emplace_back( row, col, it.value() );
            }
        }
    }

    L.setFromTriplets( triplets.begin(), triplets.end() );

    b = L.transpose() * b;
    L = L.transpose() * L;

    return { L, b };
}

}

int main( int argc, char** argv ) {
    auto opts = parseopts( argc, argv );

    std::vector< filehandle > filehandles;
    std::vector< geometry > geometries;
    for( const auto& file : opts.files ) {
        filehandles.push_back( openfile( file ) );
        geometries.push_back( findgeometry( filehandles.back().get(),
                                            opts.ilbyte,
                                            opts.xlbyte )
                            );
    }

    const auto samples = geometries.back().samples;

    const auto B = normalized_bspline( samples,
                                       opts.timeshift_resolution,
                                       3 );

    const auto C = constraints( B,
                                opts.vertical_smoothing,
                                opts.horizontal_smoothing );

    const auto Bnn = getBnn( B, opts.horizontal_smoothing );

    const auto omega = angular_frequency( samples, 1.0 );

    sparse< double > L_in;

    struct combination {
        segy_file* base;
        segy_file* monitor;
        int baseindex;
        int monindex;
    };
    std::vector< combination > vintagepairs;
    int baseindex = 0;
    for( const auto& base : filehandles ) {
        int monitorindex = baseindex + 1;
        auto monitr = filehandles.begin() + monitorindex;
        for( ; monitr != filehandles.end(); ++monitr ) {
            vintagepairs.push_back( {
                base.get(),
                monitr->get(),
                baseindex,
                monitorindex,
            } );
            ++monitorindex;
        }
        ++baseindex;
    }

    for( const auto& vp : vintagepairs ) {
        auto pair = L_ij( vp.base,
                          vp.monitor,
                          geometries.back(),
                          B,
                          C,
                          Bnn,
                          omega,
                          opts.normalizer );

        //auto& Lsquared = pair.first;
        //auto& bsquared = pair.second;

        matrix< int > maskL(filehandles.size() - 1, filehandles.size() - 1);
        maskL.setZero();
        maskL.block( vp.baseindex, vp.baseindex,
                     vp.monindex - vp.baseindex, vp.monindex - vp.baseindex )
                    .setOnes();

        // TODO: extract triplets from Lsquared; insert in L_in
    }
}
