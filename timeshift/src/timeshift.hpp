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

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

#include <segyio/segyio.hpp>

#include "linalg.hpp"

#define EIGEN_DONT_PARALLELIZE

using input_file = segyio::basic_volume< segyio::readonly >;
using output_file = segyio::basic_unstructured< segyio::trace_writer,
                                                segyio::write_always >;

namespace {

struct options {
    std::vector< std::string > files;
    std::vector< std::string > output_files;
    double      timeshift_resolution = 0.05;
    double      horizontal_smoothing = 0.01;
    double      vertical_smoothing   = 0.1;
    int         solver_max_iter      = 100;
    bool        double_precision     = false;
    bool        correct_4d_noise     = false;
    bool        cumulative           = false;
    double      scaling              = 30.0;
    double      normalization        = 0.0;
    double      sampling_interval    = 0.0;
    bool        output_norm          = false;
    bool        compute_norm         = false;
    std::string dir                  = "";
    std::string prefix               = "timeshift";
    std::string delim                = "-";
    std::string extension            = "sgy";
    int         verbosity            = 0;
    segyio::ilbyte ilbyte            = segyio::ilbyte();
    segyio::xlbyte xlbyte            = segyio::xlbyte();
};

double infer_interval( input_file& f,  double parsed ) {
    if( parsed > 0)
        return parsed;

    double fallback = 4.0;
    double inferred = 0;
    std::string fallback_msg = ", falling back to sampling interval = 4.0 ms\n";
    try {
        // TODO: use get_bin() when binary_header_reader trait is merged in segyio
        inferred = (double)f.get_th( 0 ).sample_interval / 1000.0;
    }
    catch( std::runtime_error &e ) {
        std::cout << e.what() << fallback_msg;
    }

    if( inferred <= 0 )
        std::cerr << fallback_msg;
        return fallback;

    return inferred;
}

struct Progress {
    static int expected;
    static int count;

    static void report() {
#ifndef MUTE_PROGRESS
        count++;
        if( count % (expected/20) == 0 )
            std::cout << "Progress: " << (count*100)/expected << "%\n";
#endif
    }

    static void report( int n ) {
#ifndef MUTE_PROGRESS
        for( int i = 0; i < n; ++i ) report();
#endif
    }
};

int Progress::count = 0;
int Progress::expected = 60;

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
template< typename T >
matrix< T > bspline_matrix( int n, const T* knotv, int knotlen, int order ) {
    using array = Eigen::Array< T, Eigen::Dynamic, 1 >;

    auto samples = []( int n, int order ) {
        const T interval = 1.0 / (n - 1.0);
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

    auto padknots = []( int n, const T* knotv, int knotlen, int order ) {
        const T interval = 10.0 / (n - 1.0);

        const auto padlen = order + 1;
        std::vector< T > knots( knotlen + 2*padlen );

        for( int i = 0; i <= padlen; ++i )
            knots[i] = -interval * (order - i);

        const auto scale = std::accumulate(
            knotv,
            knotv + knotlen,
            false,
            []( bool acc, T x ) { return acc or x <= 0 or 1 <= x; }
        );

        const auto denom = scale ? n + 1 : 1;

        std::transform( knotv,
                        knotv + knotlen,
                        knots.begin() + padlen,
                        [=]( T x ) { return x / denom; }
        );

        for( int i = 0; i < padlen; ++i )
            knots[padlen + knotlen + i] = 1.0 + i*interval;

        return knots;
    };

    const auto t = samples( n, order );
    const auto knots = padknots( n, knotv, knotlen, order );

    matrix< T > P = matrix< T >::Zero( t.size(), knots.size() - 1 );

    for( int i = 0; i < int(knots.size()) - 1; ++i ) {
        const auto low = knots[ i ];
        const auto high = knots[ i + 1 ];
        P.col(i) = (low <= t && t < high).template cast< T >();
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

struct geometry {
    std::size_t samples;
    std::size_t traces;
    std::size_t fast;
    std::size_t slow;

    bool operator != (const geometry& rhs) const noexcept (true) {
        return this->samples != rhs.samples
           or  this->traces  != rhs.traces
           or  this->fast  != rhs.fast
           or  this->slow  != rhs.slow
        ;
    }
};

geometry findgeometry( input_file& f ) {
    geometry geo;

    geo.samples = f.samplecount();
    geo.fast  = f.inlinecount();
    geo.slow  = f.crosslinecount();
    geo.traces  = geo.fast*geo.slow;

    if( f.sorting() == segyio::sorting::xline() )
        std::swap( geo.fast, geo.slow );

    return geo;
}

template< typename T >
T normalize_surveys( T scaling,
                     std::vector< input_file >& surveys ) {

    const auto nonzero = []( T x ) { return x != 0.0; };
    const auto abs = []( T x ) { return std::abs( x ); };
    T acc = 0;
    std::vector< T > trace( surveys.front().samplecount() );
    for( auto& survey : surveys ) {
        T sum = 0.0, count = 0.0;
        for( std::size_t trc = 0; trc < survey.tracecount(); ++trc ) {
            survey.get( trc, trace.begin() );
            std::transform( trace.begin(), trace.end(), trace.begin(), abs );
            sum   += std::accumulate( trace.begin(), trace.end(), 0.0 );
            count += std::count_if( trace.begin(), trace.end(), nonzero );
        }
        acc += sum / count;
    }

    return (acc * scaling) / surveys.size();
}

template< typename T >
void reconstruct( const Eigen::Ref< const vector< T > >& c,
                  vector< T >& reconstructed,
                  double sampling_interval,
                  const matrix< T >& B ) {

    reconstructed = sampling_interval * B * c;
}

template< typename T >
void output_timeshift( const std::string& basefile,
                       const vector< T >& coeffisients,
                       const std::string& fname,
                       const geometry& geo,
                       double sampling_interval,
                       const matrix< T >& B ) {

    std::ifstream in( basefile );
    std::ofstream dst( fname );
    dst << in.rdbuf();
    dst.close();

    output_file f( segyio::path{ fname } );

    vector< T > reconstructed( geo.samples );
    for( std::size_t traceno = 0; traceno < geo.traces; ++traceno ) {
        auto segment = coeffisients.segment( traceno * B.cols(), B.cols() );
        reconstruct< T >( segment, reconstructed, sampling_interval, B );
        f.put( traceno, reconstructed.data() );
    }
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
void derive( vector< T >& signal, const vector< T >& omega ) {

    /*
     * D = FFTI(iωFFT(signal))
     * where D is the derivative of a signal (data trace)
     *
     * It takes the entire omeg an argument, in order to not recompute it on
     * every invocation (it's shared across all derivations in this
     * application).
     */

    // TODO: MUST be initialised outside
    static std::mutex lock;
    std::lock_guard< std::mutex > guard( lock );

    static Eigen::FFT< T > fft;
    static vector< std::complex< T > > ff;

    fft.fwd( ff, signal );
    ff.array() *= std::complex< T >(0, 1) * omega.array();
    fft.inv( signal , ff );
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
T* begin( vector< T >& xs ) {
    return xs.data();
}

template< typename T >
const T* begin( const vector< T >& xs ) {
    return xs.data();
}

template< typename T >
T* end( vector< T >& xs ) {
    return xs.data() + xs.size();
}
template< typename T >
const T* end( const vector< T >& xs ) {
    return xs.data() + xs.size();
}

template< typename T >
void apply_timeshift( vector< T >& d, const vector< T >& shift ) {
    vector< T > xs = vector< T >::LinSpaced( d.size(), 0, d.size() - 1 );

    const T d_0 = d[0];
    const T d_1 = d[1];
    const T d_second_last = d[d.size() - 2];
    const T d_last = d[d.size() - 1];

    boost::math::interpolators::cardinal_cubic_b_spline< T > s(
        begin(d), end(d),
        0,
        xs[1] - xs[0]
    );

    auto interpolate = [&](T x) {
        if(x >= 0 and x <= d.size() - 1) {
            return s(x);
        }
        else if(x < 0) {
            return d_0 + (d_1 - d_0) * x;
        }
        else {
            return d_last + (d_last - d_second_last) * (x - (d.size() - 1));
        }
    };

    xs += shift;

    std::transform( begin(xs), end(xs), begin(d), interpolate );
}

template< typename T >
void correct( int start, int end,
              const vector< T >& x,
              std::vector< input_file >& files,
              const sparse< T >& spline,
              const geometry& geo,
              const vector< T >& omega,
              double normalizer,
              vector< T >& correction ) {

    /*
     * Do the correction. A lot of these functions take buffers and modify them
     * in-place for performance reasons
     *
     * TODO: wordily describe 4D correction
     */

    const int vintages = files.size();
    const std::size_t localsize = spline.cols();
    const std::size_t vintpairsize = localsize * geo.traces;

    std::vector< vector< T > > trc( vintages, vector< T >( geo.samples ) );
    std::vector< vector< T > > drv( vintages, vector< T >( geo.samples ) );

    const int timeshifts = vintages - 1;

    vector< T > sol( localsize );
    vector< T > D( geo.samples );
    vector< T > delta( geo.samples );
    vector< T > mean( geo.samples );
    vector< T > c( geo.samples );
    vector< T > shift( geo.samples );

    for( std::size_t t = start; t < end; ++t ) {

        mean.setZero();
        c.setZero();
        for( int i = 0; i < timeshifts; ++i ) {
            // TODO: comment on why
            const std::size_t r = (t * localsize) + (i * vintpairsize);
            c += spline * x.segment( r, localsize );
            mean += c;
        }
        mean /= timeshifts + 1;

        shift.setZero();
        shift -= mean;
        for( int i = 0; i < vintages; ++i ){
            auto& file = files[i];
            auto& trace = trc[i];
            auto& derived = drv[i];

            file.get( t, begin( trace ) );
            derived = (trace /= normalizer);
            // derived is updated in-place
            derive( derived, omega );

            if( i != 0 ) {
                const int r = t * localsize + (i-1) * vintpairsize;
                shift += spline * x.segment( r, localsize );
            }

            // trace is updated in-place
            apply_timeshift( trace, shift );
        }

        for( int vint1 = 0;       vint1 < vintages; ++vint1 ) {
        for( int vint2 = vint1+1; vint2 < vintages; ++vint2 ) {

            // TODO: alloc outside of loop
            const auto maskb = mask_solution( vintages, vint1, vint2 );

            D = 0.5 * ( drv[vint1] + drv[vint2] );

            delta = trc[vint1] - trc[vint2];
            // TODO: alloc outside of solution
            sol = solution( D, delta, spline );

            // TODO: rename mvrow to indicate position (maybe) in system
            for( int mvrow = 0; mvrow < timeshifts; ++mvrow ) {
                const std::size_t row = (mvrow * vintpairsize) + (t * localsize);
                correction.segment( row, localsize ) += sol * maskb( mvrow );
            }
        }}
    }

}

template< typename T >
vector< T > timeshift_4D_correction( const vector< T >& x,
                                     std::vector< input_file >& files,
                                     const sparse< T >& spline,
                                     const geometry& geo,
                                     const vector< T >& omega,
                                     double normalizer ) {

    vector< T > cs( x.size() );
    cs.setZero();

    int nthreads = 3;

    # pragma omp parallel for
    for( int thread_id = 0; thread_id < nthreads; ++thread_id ) {
        /* copy so each thread has its own file handle and I/O buffer */
        auto f = files;

        const std::size_t start = thread_id * (geo.traces/nthreads);
        const std::size_t end = (thread_id+1) == nthreads
                              ? geo.traces
                              : (thread_id+1) * (geo.traces/nthreads);

        correct( start, end, x, f, spline, geo, omega, normalizer, cs );
    }

    return cs;
}

template< typename T >
struct Si4tiPreconditioner {

    Si4tiPreconditioner() = default;

    template<typename MatrixType>
    void initialize( const MatrixType& m ) {
        this->mat = &m.mat;
        this->vintages = m.vintages;
        this->diagonals = m.diagonals;
        this->traces = m.slow*m.fast;
        this->dims = m.Bnn.rows();
    }

    template<typename MatrixType>
    Si4tiPreconditioner& analyzePattern( const MatrixType& ) {
        return *this;
    }

    template<typename MatrixType>
    Si4tiPreconditioner& factorize( const MatrixType& ) {
        return *this;
    }

    template<typename MatrixType>
    Si4tiPreconditioner& compute( const MatrixType& ) {
        return *this;
    }

    template<typename Rhs>
    inline const Rhs solve(const Eigen::MatrixBase<Rhs>& b) const {
        eigen_assert( mat
                   && "Si4tiPreconditioner is not initialized.");
        Rhs v( b.rows() );
        v.setZero();
        const int timeshifts = vintages - 1;
        const std::size_t vintpairsize = b.rows() / timeshifts;

        # pragma omp parallel for schedule(guided)
        for( std::size_t trace = 0; trace < traces; ++trace ) {

        for( int i = 0; i < timeshifts; ++i ){
            const auto col = i * diagonals;
            const auto row = i * vintpairsize + trace*dims;

            v.segment( row, dims ) += this->mat->col( col )
                                        .segment( row, dims )
                                        .cwiseInverse()
                                        .cwiseProduct( b.segment(row, dims) );
        }
        }
        return v;
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }

    const matrix< T >* mat = nullptr;
    int vintages = 0;
    int diagonals = 0;
    std::size_t traces = 0;
    std::size_t dims = 0;
};

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
                                 std::vector< input_file >& files,
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
    const int timeshifts = vintages - 1;
    const std::size_t localsize = B.cols();
    const std::size_t vintpairsize = localsize * geo.traces;
    const std::size_t solutionsize = vintpairsize * (vintages - 1);

    linear_system< T > p = {
        matrix< T >( solutionsize, ndiagonals * (vintages - 1) ),
        vector< T >( solutionsize )
    };

    p.L.setZero();
    p.b.setZero();

    auto f = files;

    std::vector< vector< T > > trc( vintages, vector< T >( geo.samples ) );
    std::vector< vector< T > > drv( vintages, vector< T >( geo.samples ) );

    vector< T > D( geo.samples );
    vector< T > delta( geo.samples );
    // TODO: sparse localL?
    matrix< T > localL( localsize, localsize );

    int processed = 0;

    # pragma omp parallel for schedule(guided) \
                 firstprivate(f, trc, drv, D, delta, localL)
    for( auto traceno = 0; traceno < geo.traces; ++traceno ) {

        if( omp_get_thread_num() == 0 ) {
            const std::size_t chunk = geo.traces / omp_get_num_threads();
            processed++;
            if( chunk > 40 && processed % (chunk/40) == 0 )
                Progress::report();
        }

        for( int i = 0; i < vintages; ++i ){
            f[i].get( traceno, trc[i].data() );
            drv[i] = trc[i] /= normalizer;
            derive( drv[i], omega );
        }

        // combinations(0..vintages)
        for( int vint1 = 0;       vint1 < vintages; ++vint1 )
        for( int vint2 = vint1+1; vint2 < vintages; ++vint2 ) {

            const auto maskL = mask_linear( vintages, vint1, vint2 );
            const auto maskb = mask_solution( vintages, vint1, vint2 );

            delta = trc[vint1] - trc[vint2];
            D = 0.5 * ( drv[vint1] + drv[vint2] );
            // TODO: if C and linearoperator(D,B) always share non-zero
            // pattern, this is reduced to an element-wise sum, and there's no
            // need to allocate on every iteration
            localL = linearoperator( D, B ) + C;

            for( int mvrow = 0; mvrow < timeshifts; ++mvrow) {
                const std::size_t row = (mvrow * vintpairsize) + (traceno * localsize);
                p.b.segment( row, localsize )
                    += solution( D, delta, B ) * maskb(mvrow);
            }

            for( int mvrow = 0; mvrow < timeshifts; ++mvrow) {
                const std::size_t row = (mvrow * vintpairsize) + (traceno * localsize);

                for( int mvcol = 0; mvcol < timeshifts; ++mvcol) {
                    if( not maskL( mvrow, mvcol ) ) continue;

                    for( int diag = 0; diag < ndiagonals; ++diag ) {
                        std::size_t col_size = localsize - diag;
                        std::size_t col = (mvcol * ndiagonals) + diag;

                        p.L.block( row, col, col_size, 1 )
                            += localL.diagonal(diag);
                    }
                }
            }
        }
    }

    return p;
}

template< typename T >
void accumulate_timeshifts( vector< T >& x, int vintages ) {
    const int timeshifts = vintages - 1;
    const std::size_t len = x.size() / timeshifts;
    for( std::size_t prev = 0, next = 1; next < timeshifts; ++next, ++prev )
        x.segment( next*len, len ) += x.segment( prev*len, len );
}

template< typename T >
vector< T > compute_timeshift( const sparse< T >& B,
                               int splineord,
                               std::vector< input_file >& files,
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

    const auto& geo = geometries.back();
    const int ndiagonals = splineord + 1;
    const auto vintages = files.size();

    const T normalizer = opts.normalization == 0.0
                            ? normalize_surveys( opts.scaling, files )
                            : opts.normalization;

    if( opts.output_norm )
        std::cout << "Normalization: " << normalizer << "\n";

    Progress::report( 5 );

    auto linear_system = build_system( B,
                                       C,
                                       omega,
                                       normalizer,
                                       files,
                                       geo,
                                       ndiagonals);

    BlockBandedMatrix< T, Progress > rep( std::move( linear_system.L ),
                                          ndiagonals,
                                          Bnn,
                                          multiplier( vintages ),
                                          vintages,
                                          geo.fast,
                                          geo.slow );

    Eigen::ConjugateGradient< decltype( rep ),
                              Eigen::Lower | Eigen::Upper,
                              Si4tiPreconditioner<T>
    > cg;
    cg.preconditioner().initialize( rep );
    cg.setMaxIterations( opts.solver_max_iter );
    cg.compute( rep );
    vector< T > x = cg.solve( linear_system.b );

    if( opts.correct_4d_noise ) {
        linear_system.b = timeshift_4D_correction( x,
                                                   files,
                                                   B,
                                                   geo,
                                                   omega,
                                                   normalizer );
        Progress::report( 20 );
        x -= cg.solve( linear_system.b );
    }

    if( opts.cumulative ) {
        accumulate_timeshifts( x, vintages );
    }

    return x;
}

}
