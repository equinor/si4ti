#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <mutex>

#include <getopt.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>
#include "spline.h"

#define EIGEN_DONT_PARALLELIZE

#include <segyio/segyio.hpp>

namespace {

struct options {
    std::vector< std::string > files;
    double timeshift_resolution = 0.05;
    double horizontal_smoothing = 0.01;
    double vertical_smoothing   = 0.1;
    int    solver_max_iter      = 100;
    bool   double_precision     = false;
    bool   correct_4d_noise     = false;
    bool   cumulative           = false;
    double datascaling          = 30;
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
        { "cumulative",           no_argument,       0, 's' },
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
                             "r:H:V:m:dcsn:i:x:v",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'r': break;
            case 'H': opts.horizontal_smoothing = std::stod( optarg ); break;
            case 'V': opts.vertical_smoothing   = std::stod( optarg ); break;
            case 'm': opts.solver_max_iter      = std::stoi( optarg ); break;
            case 'd': break;
            case 'c': opts.correct_4d_noise     = true; break;
            case 's': opts.cumulative           = true; break;
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

struct geometry {
    int samples;
    int traces;
    int ilines;
    int xlines;
};

geometry findgeometry( sio::simple_file f ) {
    geometry geo;

    geo.samples = f.read(0).size();
    geo.ilines  = f.inlines().size();
    geo.xlines  = f.crosslines().size();
    geo.traces  = geo.ilines*geo.xlines;

    return geo;
}

template< typename T >
T normalize_surveys( const T x,
                     std::vector< sio::simple_file >& surveys ) {
    T acc = 0;
    std::vector< T > trace;
    for( auto& survey : surveys ) {
        T sum = 0, count = 0;
        for( int trc = 0; trc < survey.size(); ++trc ) {
            survey.read( trc, trace );
            std::transform(trace.begin(), trace.end(), trace.begin(), fabs);
            sum   += std::accumulate(trace.begin(), trace.end(), 0.0);
            count += std::count_if( trace.begin(), trace.end(),
                                    []( T i ) { return i > 0; });
        }
        acc += sum / count;
    }
    return (acc * x) / surveys.size();
}

template< typename Vector >
void writefile( const std::string& basefile,
                const Vector& v,
                const std::string& fname,
                const geometry& geo ) {

    std::ifstream in( basefile );
    std::ofstream dst( fname );
    dst << in.rdbuf();
    dst.close();

    sio::simple_file f( fname, sio::config().readwrite() );

    auto itr = v.data();
    for( int traceno = 0; traceno < geo.traces; ++traceno )
        itr = f.put( traceno, itr );
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

    for( auto f = middle + step; f <= samples - 1/density; f += step )
        knotv.push_back( f );

    return knotv;
}

template< typename T >
sparse< T > normalized_bspline( int samples, T density, int degree ) {
    const auto knotv = knotvector( samples, density );
    auto B = bspline_matrix( samples, knotv.data(), knotv.size(), degree );
    return (B * B.colwise().sum().cwiseInverse().asDiagonal()).sparseView();
}

template< typename T >
sparse< T > constraints( const sparse< T >& spline,
                         double vertical_smoothing,
                         double horizontal_smoothing ) {

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
    sparse< T > Dpm = (spline.transpose() * D.transpose() * D * spline).sparseView();

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
sparse< T > linearoperator( const vector< T >& derived,
                            const sparse< T >& spline ) {
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
               const sparse< T >& spline ) -> vector< T > {
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

template< typename T >
sparse< T > getBnn( const sparse< T >& B,
                    double horizontal_smoothing ) {
    return (B.transpose() * B) * .25 * horizontal_smoothing;
}

matrix< int > mask_linear( int vintages, int baseindex, int monindex ) {
    // TODO: doc masks
    matrix< int > maskL( vintages - 1, vintages - 1 );
    maskL.setZero();

    const auto masksize = monindex - baseindex;
    maskL.block( baseindex, baseindex, masksize, masksize ).setOnes();
    return maskL;
}

matrix< int > multiplier( int vintages ) {
    matrix< int > mp( vintages - 1, vintages - 1 );
    mp.setZero();

    for( int i = 0;   i < vintages; ++i ) {
    for( int j = i+1; j < vintages; ++j ) {
        mp += mask_linear( vintages, i, j );
    }}

    return mp;
}

vector< int > mask_solution( int vintages, int baseindex, int monindex ) {
    vector< int > maskb( vintages - 1 );
    maskb.setZero();

    const auto masksize = monindex - baseindex;
    maskb.segment( baseindex, masksize ).setOnes();
    return maskb;
}

template< typename T >
vector< T >& shift_data( vector< T >& d, const vector< T >& corr ) {
    vector< T > linspace = vector< T >::LinSpaced( d.size(), 0, d.size() - 1 );

    std::vector< double > ls( linspace.data(), linspace.data() + linspace.size());
    std::vector< double > dd( d.data(), d.data() + d.size());
    tk::spline s;
    s.set_points( ls, dd );

    linspace -= corr;

    for( int i = 0; i < d.size(); ++i )
        d(i) = s( linspace(i) );
    return d;
}

template< typename T >
vector< T > timeshift_4D_correction( const vector< T >& x,
                                     std::vector< sio::simple_file> files,
                                     const sparse< T >& spline,
                                     const geometry& geo,
                                     const vector< T >& omega,
                                     const double normalizer ) {

    const int vintages = files.size();
    const int localsize = spline.cols();
    const int vintpairsize = localsize * geo.traces;

    vector< T > corr( x.size() );
    corr.setZero();

    std::mutex derive_lock;

    int nthreads = 3;

    # pragma omp parallel for
    for( int tnr = 0; tnr < nthreads; ++tnr )
    {
    std::vector< sio::simple_file > f( files );

    std::vector< vector< T > > trc( vintages, vector< T >( geo.samples ) );
    std::vector< vector< T > > drv( vintages, vector< T >( geo.samples ) );

    vector< T > sol( localsize );
    vector< T > D( geo.samples );
    vector< T > delta( geo.samples );
    vector< T > mean( geo.samples );
    vector< T > shift( geo.samples );

    const int start = tnr * (geo.traces/nthreads);
    const int end = (tnr+1) == nthreads ? geo.traces
                                        : (tnr+1) * (geo.traces/nthreads);

    for( int t = start; t < end; ++t ) {

        mean.setZero();
        for( int i = 0; i < vintages-1; ++i ) {
            const int r = (t * localsize) + (i * vintpairsize);
            mean += mean + spline * x.segment( r, localsize );
        }
        mean /= vintages-1;

        for( int i = 0; i < vintages; ++i ){
            f[i].read( t, trc[i].data() );
            trc[i] /= normalizer;
            D = trc[i];
            derive_lock.lock();
            drv[i] = derive( D, omega );
            derive_lock.unlock();

            shift.setZero();
            for( int j = 0; j < i; ++j ) {
                const int r = (t * localsize) + (j * vintpairsize);
                shift += spline * x.segment( r, localsize );
            }
            shift -= mean;
            shift_data( trc[i], shift );
        }

        for( int vint1 = 0;       vint1 < vintages; ++vint1 ) {
        for( int vint2 = vint1+1; vint2 < vintages; ++vint2 ) {

            const auto maskb = mask_solution( vintages, vint1, vint2 );

            D = 0.5 * ( drv[vint1] + drv[vint2] );

            delta = trc[vint2] - trc[vint1];
            sol = solution( D, delta, spline );

            for( int mvrow = 0; mvrow < vintages - 1; ++mvrow ) {
                const int row = (mvrow * vintpairsize) + (t * localsize);
                corr.segment( row, localsize )
                    += sol * maskb(mvrow);
            }
        }}
    }
    }
    return corr;
}

template< typename T > class SuperMatrix;

template< typename T >
struct SuperMatrix : public Eigen::EigenBase< SuperMatrix< T > > {
    using Scalar = T;
    using RealScalar = T;
    using StorageIndex = std::ptrdiff_t;
    using Index = typename Eigen::EigenBase< SuperMatrix >::Index;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    SuperMatrix( matrix< T > m,
                 int diagonals,
                 sparse< T > bnn,
                 matrix< int > cmb,
                 int vints,
                 int inlines,
                 int crosslines ) :
        mat( std::move( m ) ),
        diagonals( diagonals ),
        Bnn( std::move( bnn ) ),
        comb( std::move( cmb ) ),
        vintages( vints ),
        ilines( inlines ),
        xlines( crosslines )
    {}

    Index rows() const { return this->mat.rows(); }
    Index cols() const { return this->mat.rows(); }

    template< typename Rhs >
    Eigen::Product< SuperMatrix, Rhs, Eigen::AliasFreeProduct >
    operator*( const Eigen::MatrixBase< Rhs >& x ) const {
        return { *this, x.derived() };
    }

    matrix< T > mat;
    int diagonals;
    sparse< T > Bnn;
    matrix< int > comb;
    int vintages;
    int ilines, xlines;
};

template< typename T >
struct SuperPreconditioner {

    SuperPreconditioner() = default;

    template<typename MatrixType>
    void initialize( const MatrixType& m ) {
        this->mat = &m.mat;
        this->vintages = m.vintages;
        this->diagonals = m.diagonals;
        this->traces = m.xlines*m.ilines;
        this->dims = m.Bnn.rows();
    }

    template<typename MatrixType>
    SuperPreconditioner& analyzePattern( const MatrixType& m ) {
        return *this;
    }

    template<typename MatrixType>
    SuperPreconditioner& factorize( const MatrixType& m ) {
        return *this;
    }

    template<typename MatrixType>
    SuperPreconditioner& compute( const MatrixType& m ) {
        return *this;
    }

    template<typename Rhs>
    inline const Rhs solve(const Eigen::MatrixBase<Rhs>& b) const {
        eigen_assert( !mat
                   && "SuperPreconditioner is not initialized.");
        vector< T > v( b.rows() );
        v.setZero();
        const int vintpairsize = b.rows() / (vintages - 1);

        # pragma omp parallel
        {
        const int nthreads = omp_get_num_threads();
        const int tnr = omp_get_thread_num();

        const int start = tnr * (traces/nthreads);
        const int end = (tnr+1) == nthreads ? traces
                                            : (tnr+1) * (traces/nthreads);

        const int len = (end-start)*dims;

        for( int i = 0; i < vintages - 1; ++i ){
            const auto col = i * diagonals;
            const auto row = i * vintpairsize + start*dims;
            v.segment( row, len ).array()
              += this->mat->col( col )
                          .segment( row, len )
                          .cwiseInverse()
                          .array()
               * b.segment( row, len )
                  .array();
        }
        }
        return v;
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }

    const matrix< T >* mat = nullptr;
    int vintages;
    int diagonals;
    int traces;
    int dims;
};

}

namespace Eigen { namespace internal {

template<>
template< typename T >
struct traits< SuperMatrix< T > > :
    public Eigen::internal::traits< Eigen::SparseMatrix< T > >
{};

template< typename T, typename Rhs >
struct generic_product_impl< SuperMatrix< T >,
                             Rhs,
                             SparseShape,
                             DenseShape,
                             GemvProduct // GEMV stands for matrix-vector
                           >
     : generic_product_impl_base< SuperMatrix< T >,
                                  Rhs,
                                  generic_product_impl< SuperMatrix< T >, Rhs >
                                >
{
    using Scalar = typename Product< SuperMatrix< T >, Rhs >::Scalar;
    template< typename Dest >
    static void scaleAndAddTo( Dest& dst,
                               const SuperMatrix< T >& lhs,
                               const Rhs& rhs,
                               const Scalar& alpha ) {

        const auto ilines     = lhs.ilines;
        const auto xlines     = lhs.xlines;
        const auto traces     = ilines * xlines;
        const auto localsize  = lhs.Bnn.rows();

        # pragma omp parallel
        {
        const int nthreads = omp_get_num_threads( );
        const int tnr = omp_get_thread_num( );

        const int start = tnr * (traces/nthreads);
        const int end = (tnr+1) == nthreads ? traces
                                            : (tnr+1) * (traces/nthreads);

        const auto vintages = lhs.vintages;
        const auto diagonals = lhs.diagonals;
        const auto vintpairsize = lhs.mat.rows() / (vintages - 1);

        for( int mvrow = 0; mvrow < vintages-1 ; ++mvrow ) {
        for( int mvcol = 0; mvcol < vintages-1 ; ++mvcol ) {
        for( int diag = 0; diag < diagonals; ++diag ) {
            const auto lhs_col   = mvcol * diagonals + diag;
            const auto lhs_start = mvrow * vintpairsize + start*localsize;
            const auto dst_start = mvrow * vintpairsize + start*localsize;
            const auto len       =
                (tnr+1) == nthreads ? (end-start)*localsize - diag
                                    : (end-start)*localsize;
            const auto rhs_start =
                diag + mvcol * vintpairsize + start*localsize;

            dst.segment( dst_start, len )
               .array()
                += alpha * (lhs.mat.col( lhs_col ).segment( lhs_start, len )
                                                  .array()
                         * rhs.segment( rhs_start, len )
                              .array());
            #pragma omp barrier
            if( diag > 0 )
                dst.segment( rhs_start, len )
                   .array()
                    += alpha * (lhs.mat.col( lhs_col ).segment( lhs_start, len )
                                                      .array()
                             * rhs.segment( dst_start, len )
                                  .array());
            #pragma omp barrier

        }}}

        const auto vintsize = localsize * traces;
        const auto& comb    = lhs.comb;

        for( int vint1 = 0; vint1 < vintages - 1; ++vint1 ) {
        for( int vint2 = 0; vint2 < vintages - 1; ++vint2 ) {

        int j = start / xlines;
        int k = start % xlines;
        for( int i = start; i < end; ++i ) {
            const std::ptrdiff_t iss[] = {
                // C(j-1, k)
                k == 0 ? i : i - 1,
                // C(j+1, k)
                k == xlines - 1 ? i : i + 1,
                // C(j, k-1)
                j == 0 ? i : i - xlines,
                // C(j, k+1)
                j == ilines - 1 ? i : i + xlines,
            };

            // move one xline forward, unless we're at the last xline in this
            // inline, then wrap around to zero again
            k = k + 1 >= xlines ? 0 : k + 1;
            // move the ilines forwards every time xlines wrap around
            if( k == 0 ) ++j;

            const auto col = (vint1 * vintsize) + i * localsize;
            for( const auto is : iss ) {
                const auto row = (vint2 * vintsize) + is * localsize;
                vector< T > x = rhs.segment( row, localsize );
                vector< T > smoothing = comb(vint1, vint2) * (lhs.Bnn * x).eval();
                dst.segment(col, localsize).array() -= smoothing.array();
            }
        }

        }}
        }
    }
};

} }

#ifndef TEST

template< typename T >
struct linear_system {
    matrix< T > L;
    vector< T > b;
};

template< typename T >
linear_system< T > build_system( const sparse< T >& B,
                                 const sparse< T >& C,
                                 const vector< T >& omega,
                                 double normalizer,
                                 std::vector< sio::simple_file >& files,
                                 const geometry& geo,
                                 const int ndiagonals) {

    /*
     * Builds the linear system (L, b and multiplier).
     *
     * The matrix L consists of banded, symmetric matrix blocks and can thus be
     * stored very efficiently by only storing the uppermpost non-zero
     * diagonals.
     *
     *   a b 0 0 .. c d 0 0       a b .. c d
     *   b e f 0 .. d g h 0       e f .. g h
     *   0 f i j .. 0 h k l       i j .. k l
     *   0 0 j m .. 0 0 l n       m 0 .. n 0
     *   : : : : :: : : : :  -->  : : :: : :
     *   o p 0 0 .. q r 0 0       o p .. q r
     *   n s t 0 .. r u v 0       s t .. u v
     *   0 t w x .. 0 v y z       w x .. y z
     *   0 0 x æ .. 0 0 z ø       æ 0 .. ø 0
     */

    const int vintages = files.size();
    const int localsize = B.cols();
    const int vintpairsize = localsize * geo.traces;
    const int solutionsize = vintpairsize * (vintages - 1);

    linear_system< T > p = {
        matrix< T >( solutionsize, ndiagonals * (vintages - 1) ),
        vector< T >( solutionsize )
    };

    p.L.setZero();
    p.b.setZero();

    std::mutex derive_lock;

    int nthreads = 3;

    # pragma omp parallel for
    for( int tnr = 0; tnr < nthreads; ++tnr )
    {

    std::vector< sio::simple_file > f( files );

    std::vector< vector< T > > trc( vintages, vector< T >( geo.samples ) );
    std::vector< vector< T > > drv( vintages, vector< T >( geo.samples ) );

    vector< T > D( geo.samples );
    vector< T > delta( geo.samples );
    matrix< T > localL( localsize, localsize );
    vector< T > localb( localsize );

    const int start = tnr * (geo.traces/nthreads);
    const int end = (tnr+1) == nthreads ? geo.traces
                                        : (tnr+1) * (geo.traces/nthreads);

    for( auto traceno = start; traceno < end; ++traceno ) {

        for( int i = 0; i < vintages; ++i ){
            f[i].read( traceno, trc[i].data() );
            trc[i] /= normalizer;
            D = trc[i];
            derive_lock.lock();
            drv[i] = derive( D, omega );
            derive_lock.unlock();
        }

        for( int vint1 = 0;       vint1 < vintages; ++vint1 ) {
        for( int vint2 = vint1+1; vint2 < vintages; ++vint2 ) {

            const auto maskL = mask_linear( vintages,
                                            vint1,
                                            vint2 );
            const auto maskb = mask_solution( vintages,
                                              vint1,
                                              vint2 );

            delta = trc[vint2] - trc[vint1];
            D = 0.5 * ( drv[vint1] + drv[vint2] );
            localL = linearoperator( D, B ) + C;

            for( int mvrow = 0; mvrow < vintages - 1; ++mvrow) {
                int row = (mvrow * vintpairsize) + (traceno * localsize);
                p.b.segment( row, localsize )
                    += solution( D, delta, B ) * maskb(mvrow);

                for( int mvcol = 0; mvcol < vintages - 1; ++mvcol) {
                    if( maskL( mvrow, mvcol ) ){
                        for( int diag = 0; diag < ndiagonals; ++diag ) {

                            int col_size = localsize - diag;
                            int col = (mvcol * ndiagonals) + diag;

                            p.L.block( row, col, col_size, 1 )
                                += localL.diagonal(diag);
                        }
                    }
                }
            }

        }}
    }
    }
    return p;
}

template< typename T >
void accumulate_timeshifts( vector< T >& x, int vintages ) {
    int sz = x.size() / (vintages-1);
    for( int i = 1; i < vintages-1; ++i )
        x.segment( i*sz, sz ) += x.segment( (i-1)*sz, sz );
}

template< typename T >
vector< T > compute_timeshift( const sparse< T >& B,
                               int splineord,
                               std::vector< sio::simple_file >& files,
                               const std::vector< geometry >& geometries,
                               const options& opts ) {

    /* The reason for separating this part in a function is to trigger implicit
     * cleanup of objects. This signifficantly reduces the maximum memory
     * consumption.
     */

    const auto C = constraints( B,
                                opts.vertical_smoothing,
                                opts.horizontal_smoothing );

    const auto Bnn = getBnn( B, opts.horizontal_smoothing );

    const auto samples = geometries.back().samples;
    const auto omega = angular_frequency( samples, T( 1.0 ) );

    const auto solsize =
        B.cols() * geometries.front().traces * (files.size() - 1);

    const auto& geo = geometries.back();
    const int ndiagonals = splineord + 1;
    const auto vintages = files.size();

    const T normalizer = normalize_surveys( opts.datascaling, files );

    auto linear_system = build_system( B,
                                       C,
                                       omega,
                                       normalizer,
                                       files,
                                       geo,
                                       ndiagonals);

    SuperMatrix< T > rep( std::move( linear_system.L ),
                          ndiagonals,
                          Bnn,
                          multiplier( vintages ),
                          vintages,
                          geo.ilines,
                          geo.xlines );

    Eigen::ConjugateGradient<
        decltype( rep ),
        Eigen::Lower | Eigen::Upper,
        SuperPreconditioner<T>
    > cg;
    cg.preconditioner().initialize( rep );
    cg.setMaxIterations( opts.solver_max_iter );
    cg.compute( rep );
    vector< T > x = cg.solve( linear_system.b );

    if( opts.correct_4d_noise ) {
        linear_system.b = timeshift_4D_correction( x, files, B, geo, omega, normalizer );
        x -= cg.solve( linear_system.b );
    }

    return x;
}

int main( int argc, char** argv ) {
    //std::cout << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n";
    Eigen::initParallel();
    auto opts = parseopts( argc, argv );

    std::vector< sio::simple_file > files;
    std::vector< geometry > geometries;
    for( const auto& file : opts.files ) {
        files.push_back( { file, sio::config().ilbyte( opts.ilbyte )
                                              .xlbyte( opts.xlbyte ) }  );

        geometries.push_back( findgeometry( files.back() )
                            );
    }

    using T = float;

    const auto vintages = files.size();
    const auto samples = geometries.back().samples;
    const int splineord = 3;
    const auto B = normalized_bspline( samples,
                                       T( opts.timeshift_resolution ),
                                       splineord );

    auto x = compute_timeshift( B, splineord, files, geometries, opts );

    if( opts.cumulative )
        accumulate_timeshifts( x, vintages );

    auto reconstruct = [&]( vector< T > seg ) {
        const auto scale = 4.0;
        const auto samples = geometries.front().samples;
        const auto M = B.cols();

        vector< T > reconstructed( geometries.front().traces * samples );
        for( int i = 0; i < geometries.front().traces; ++i ) {
            reconstructed.segment( i * samples, samples )
                = scale * B * seg.segment( i * M, M );
        }

        return reconstructed;
    };

    const auto M = B.cols() * geometries.front().traces;

    for( int i = 0; i < vintages - 1; ++i ) {
        T scale = 4;
        vector< T > seg = x.segment( i * M, M );
        auto timeshift = reconstruct( seg );
        writefile( opts.files.front(),
                   timeshift,
                   "timeshift-" + std::to_string( i ) + ".sgy",
                   geometries.back() );

    }
}

#endif
