#ifndef IMPEDANCE_HPP
#define IMPEDANCE_HPP
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <ctime>
#include <utility>
#include <cmath>
#include <algorithm>

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>
#include <segyio/segyio.hpp>

using namespace segyio::literals;

namespace {

struct options {
    std::vector< std::string > files;
    std::vector< std::string > output_files;
    int             verbosity            = 0;
    segyio::ilbyte  ilbyte               = segyio::ilbyte();
    segyio::xlbyte  xlbyte               = segyio::xlbyte();
    int             polarity             = 1;
    int             segments             = 1;
    int             overlap              = -1;
    bool            tv_wavelet           = false;
    double          damping_3D           = 0.0001;
    double          damping_4D           = 0.0001;
    double          latsmooth_3D         = 0.05;
    double          latsmooth_4D         = 4;
    int             max_iter             = 50;
};


struct Progress {
    static int expected;
    static int count;

    static void report() {
        count++;
        if( count % (expected/20) == 0 )
            std::cout << "Progress: " << (count*100)/expected << "%\n";
    }

    static void report( int n ) {
        for( int i = 0; i < n; ++i ) report();
    }
};

template< typename T >
using vector = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
template< typename T >
using matrix = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
void upshift_cols( Eigen::Ref< matrix< T > > m, int shift ) {
    matrix< T > tmp = m.topRows( shift );
    for( int i = 0; i < m.rows() - shift; ++i )
        m.row( i ) = m.row( i + shift );
    m.bottomRows( shift ) = tmp;
}

template< typename T >
void ifftshift( Eigen::Ref< matrix< T > > m ) {
    return upshift_cols( m, m.rows() / 2 );
}

template< typename T >
vector< T > myhamn( int n ){
    constexpr static const T pi = 3.14159265358979323846;
    vector< T > filter;
    filter.setLinSpaced( n, 0, 2*pi );
    return 0.5 * (1 - filter.array().cos());
}

template< typename T, typename INFILE_TYPE >
vector< T > timeinvariant_wavelet( INFILE_TYPE& survey ) {

    const auto tracelen = survey.samplecount();

    vector< T > freqwav( tracelen );
    freqwav.setZero();

    vector< T > trace( tracelen );
    Eigen::FFT< T > fft;
    vector< std::complex< T > > spectrum;

    for( std::size_t trc = 0; trc < survey.tracecount(); ++trc ) {
        survey.get( trc, trace.data() );
        fft.fwd( spectrum, trace );
        freqwav += spectrum.cwiseAbs() / survey.tracecount();
    }

    return freqwav;
}

template< typename T, typename INFILE_TYPE >
matrix< T > timevarying_wavelet( INFILE_TYPE& survey ) {

    const auto tracelen = survey.samplecount();
    const int win_size = 101;
    const int win_step = 20;

    matrix< T > freqwav = matrix< T >::Zero( tracelen, tracelen );

    vector< T > trace( tracelen );
    Eigen::FFT< T > fft;
    vector< std::complex< T > > spectrum( win_size );

    const int last_offset = ((tracelen-win_size)/win_step)*win_step + win_size/2;

    for( std::size_t trc = 0; trc < survey.tracecount(); ++trc ) {
        survey.get( trc, trace.data() );
        for( int offset = win_size/2; offset <= last_offset; offset += win_step ) {
            auto segment = trace.segment( offset-win_size/2, win_size );
            fft.fwd( spectrum, segment );
            freqwav.col( offset ).head( win_size )
                += spectrum.cwiseAbs() / survey.tracecount();
        }
    }

    freqwav.leftCols( win_size/2 ).colwise() = freqwav.col( win_size/2 );
    freqwav.rightCols( tracelen - last_offset ).colwise() = freqwav.col( last_offset );

    for( int offset = win_size/2; offset < last_offset; offset += win_step ) {
        for( int i = 1; i < win_step; ++i ) {
            freqwav.col( offset + i )
                = (1-T(i)/win_step)*freqwav.col( offset )
                + (T(i)/win_step)*(freqwav.col( offset + win_step ));
        }
    }

    for( int c = 0; c < tracelen; ++c ) {
        auto segment = freqwav.col(c).head( win_size );
        spectrum = segment;
        fft.inv( segment, spectrum );
    }

    matrix< T > mh( win_size, tracelen );
    mh.colwise() = myhamn< T >( win_size );

    ifftshift< T >( freqwav.topRows( win_size ) );
    freqwav.topRows( win_size ) = mh.array() * freqwav.topRows( win_size ).array();

    vector< std::complex< T > > spectrum2( tracelen );
    for( int c = 0; c < tracelen; ++c ) {
        auto col = freqwav.col(c);
        fft.fwd( spectrum2, col );
        col = spectrum2.cwiseAbs();
    }

    freqwav.array().rowwise() /=
           freqwav.colwise().mean().array() / freqwav.mean()
         + std::numeric_limits<T>::epsilon();

    return freqwav;
}

template< typename T, typename INFILE_TYPE >
std::vector< matrix< T > > wavelets( std::vector< INFILE_TYPE >& vintages,
                                     bool tv_wavelet,
                                     int polarity ) {
    std::vector< matrix< T > > wvlets;

    for( auto& vintage : vintages ) {
        matrix< T > wavelet;
        if( tv_wavelet )
            wavelet = timevarying_wavelet< T >( vintage );
        else
            wavelet = timeinvariant_wavelet< T >( vintage );

        wavelet *= polarity;
        wvlets.emplace_back( std::move( wavelet ) );
    }

    return wvlets;
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
matrix< T > mask( int tracelen ) {
    matrix< T > msk( tracelen, tracelen );
    msk.colwise() = vector< T >::LinSpaced( tracelen, 1, tracelen );
    matrix< T > tmp = msk.transpose();
    msk -= tmp;
    msk = 1 / ( 1 + ( -msk.array() - T(tracelen)/2 ).exp() )
        - 1 / ( 1 + ( -msk.array() + T(tracelen)/2 ).exp() );

    return msk;
}

template< typename T >
matrix< T > forward_operator( const matrix< T >& wavelet ) {

    const int tracelen = wavelet.rows();

    matrix< T > D = matrix< T >::Identity( tracelen, tracelen );
    D.bottomRows( D.rows()-1 ) -= matrix< T >::Identity( tracelen-1, tracelen );

    matrix< T > msk = mask< T >( tracelen );

    constexpr static const T pi = 3.14159265358979323846;
    constexpr static const std::complex< T > i(0,1);
    vector< T > f = frequency_spectrum< T >( tracelen );
    vector< std::complex< T > > tmp( tracelen );

    Eigen::FFT< T > fft;

    matrix< T > A( tracelen, tracelen );

    for( int j = 0; j < tracelen; ++j ) {
        auto wv = wavelet.cols() == 1 ? wavelet : wavelet.col(j);

        tmp = T(-2)*i*pi * f * (j-T(0.5));
        tmp = wv.array() * tmp.array().exp();
        A.col( j ) = fft.inv( tmp );
        A.col( j ).array() *= msk.col( j ).array();
    }
    A *= D;

    return A;
}

template< typename T >
std::vector< matrix< T > > forward_operators( std::vector< matrix< T > >& wvlets,
                                              int vintages,
                                              T norm ) {
    std::vector< matrix< T > > As;

    for( int i = 0; i < vintages; ++i ) {
        matrix< T > A =
            forward_operator< T >( wvlets[ i ] ) / norm;
        As.emplace_back( std::move( A ) );
    }

    return As;
}

template< typename T >
matrix< T > inverse_operator( const matrix< T >& A, T damping ) {
    const int tracelen = A.cols();
    return A.transpose()*A + damping*matrix< T >::Identity( tracelen, tracelen );
}

template< typename T >
std::vector< matrix< T > > inverse_operators( const std::vector< matrix< T > >& A,
                                              int vintages,
                                              T damping ) {
    std::vector< matrix< T > > Ls;

    for( int v = 0; v < vintages; ++v ) {
        matrix< T > L = inverse_operator( A[v], damping );
        Ls.emplace_back( std::move( L ) );
    }

    return Ls;
}

template< typename T >
struct solution_1D {
    vector< T > b;
    vector< T > rj;
};

template< typename T, typename INFILE_TYPE >
solution_1D< T > solve_1D( std::vector< INFILE_TYPE >&  vintages,
                           const std::vector< matrix< T > >& L,
                           const std::vector< matrix< T > >& A,
                           T damping,
                           int trc_start, int trc_end ) {

    const std::size_t traces =  trc_end - trc_start + 1;
    const std::size_t tracelen = L[0].cols();
    const std::size_t cubesize = traces * tracelen;
    const int nvint = vintages.size();

    solution_1D< T > sol = {
        vector< T >( nvint * cubesize ),
        vector< T >( nvint * cubesize )
    };

    vector< T > trace( tracelen );
    const matrix< T > dmp =
        damping * matrix< T >::Identity( tracelen, tracelen );
    matrix< T > L_in( tracelen, tracelen );

    for( int v = 0; v < nvint; ++v ) {

        Progress::report( 20 / nvint );

        L_in = (L[v] + dmp).inverse();
        auto& vintage = vintages[v];

        for( std::size_t t = trc_start; t <= trc_end; ++t ) {
            const std::size_t offset = v * cubesize + (t-trc_start) * tracelen;
            vintage.get( t, trace.data() );
            sol.b.segment( offset, tracelen ) = A[v].transpose() * trace;
        }

        Eigen::Map< matrix< T > > x( sol.b.data() + v*cubesize,
                                     tracelen,
                                     traces );
        Eigen::Map< matrix< T > > y( sol.rj.data() + v*cubesize,
                                     tracelen,
                                     traces );
        y = L_in * x;
    }

    return sol;
}

template< typename T, typename OUTFILE_TYPE >
void add_boundary_inline( std::vector< OUTFILE_TYPE >& relAI_files,
                          vector< T >& b,
                          T norm,
                          T lat_smooth_3D, T lat_smooth_4D,
                          std::size_t trc_start, std::size_t trc_end ) {
    /*
     * Adds lateral 3D/4D smoothing and 4D damping contributions from the last
     * trace of the previous segment.
     */

    const int nvints = relAI_files.size();

    const bool xlinesorted =
        relAI_files.front().sorting() == segyio::sorting::xline();
    const std::size_t slow = xlinesorted ?
        relAI_files.front().inlinecount() : relAI_files.front().crosslinecount();

    const std::size_t traces = trc_end - trc_start + 1;
    const std::size_t tracelen = relAI_files.front().samplecount();
    const std::size_t cubesize = tracelen * traces;

    vector< T > trace( tracelen );
    vector< T > trace2( tracelen );

    for( int v = 0; v < nvints; ++v ) {
        for( std::size_t t = (trc_start - slow); t < trc_start; ++t ) {
            const std::size_t offset =
                v * cubesize + (t-(trc_start-slow)) * tracelen;

            relAI_files[v].get( t, trace.data() );
            b.segment( offset, tracelen ) += norm * (lat_smooth_3D/4) * trace;

            for( int v2 = 0; v2 < nvints; ++v2 ) {
                if( v == v2 ) continue;

                relAI_files[v2].get( t, trace2.data() );
                b.segment( offset, tracelen ) +=
                    norm * (lat_smooth_4D/4) * ( trace - trace2 );
            }
        }
    }
}

struct QuietReporter {
    static void report(){};
};

template< typename T, typename Reporter = QuietReporter >
struct Si4tiImpMatrix : public Eigen::EigenBase< Si4tiImpMatrix< T, Reporter > > {
    using Scalar = T;
    using RealScalar = T;
    using StorageIndex = std::ptrdiff_t;
    using Index = typename Eigen::EigenBase< Si4tiImpMatrix >::Index;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Si4tiImpMatrix( std::vector< matrix< T > > m,
                     int vints,
                     int fast,
                     int slow,
                     T damping_4D,
                     T lat_smooth_3D,
                     T lat_smooth_4D,
                     bool sgmented ) :
        mat( std::move( m ) ),
        vintages( vints ),
        fast( fast ),
        slow( slow ),
        damping_4D( damping_4D ),
        lat_smooth_3D( lat_smooth_3D ),
        lat_smooth_4D( lat_smooth_4D ),
        segmented( sgmented )
    {}

    Index rows() const {
        return this->mat[0].rows() * vintages * slow * fast;
    }

    Index cols() const {
        return this->mat[0].rows() * vintages * slow * fast;
    }

    template< typename Rhs >
    Eigen::Product< Si4tiImpMatrix, Rhs, Eigen::AliasFreeProduct >
    operator*( const Eigen::MatrixBase< Rhs >& x ) const {
        Reporter::report();
        return { *this, x.derived() };
    }

    std::vector< matrix< T > > mat;
    int vintages;
    int fast, slow;
    T damping_4D;
    T lat_smooth_3D;
    T lat_smooth_4D;
    bool segmented;

};

template< typename T, typename MatrixType >
vector< T > conjugate_gradient( const MatrixType& L,
                                vector< T >& x,
                                vector< T >& b,
                                int iterations ) {
    b = b - L * x;

    vector< T > p = b;
    vector< T > q( b.size() );
    double rho_2;

    for( int i = 0; i < iterations; ++i ) {
        double rho_1 = b.squaredNorm();

        if( i != 0 ) {
            double beta = rho_1 / rho_2;
            p = b + beta*p;
        }

        q = L * p;
        double alpha = rho_1 / p.dot(q);
        x = x + alpha * p;
        b = b - alpha * q;
        rho_2 = rho_1;
    }

    return x;
}

template< typename T, typename INFILE_TYPE, typename OUTFILE_TYPE >
vector< T > compute_impedance( std::vector< INFILE_TYPE >& vintages,
                               std::vector< OUTFILE_TYPE >& relAI_files,
                               const std::vector< matrix< T > >& A,
                               T norm,
                               int max_iter,
                               T damping_3D,
                               T damping_4D,
                               T lat_smooth_3D,
                               T lat_smooth_4D,
                               int trc_start,
                               int trc_end ) {

    bool xlinesorted = vintages.front().sorting() == segyio::sorting::xline();
    const int slow = xlinesorted ?
        vintages.front().inlinecount() : vintages.front().crosslinecount();

    const int fast = (trc_end - trc_start + 1) / slow;

    const int nvints = vintages.size();
    const bool segmented = trc_start != 0;

    std::vector< matrix< T > > L = inverse_operators< T >( A,
                                                           nvints,
                                                           damping_3D );

    solution_1D< T > sol = solve_1D( vintages,
                                     L,
                                     A,
                                     damping_3D,
                                     trc_start, trc_end );

    if( segmented ) {
        add_boundary_inline( relAI_files,
                             sol.b,
                             norm,
                             lat_smooth_3D, lat_smooth_4D,
                             trc_start, trc_end );
    }

    Si4tiImpMatrix< T, Progress > rbd_L( std::move( L ),
                                         nvints,
                                         fast,
                                         slow,
                                         damping_4D,
                                         lat_smooth_3D,
                                         lat_smooth_4D,
                                         segmented );

    conjugate_gradient( rbd_L, sol.rj, sol.b, max_iter );

    return sol.rj / norm;
}

template< typename Vector, typename OUTFILE_TYPE >
void writefile( const Vector& v,
                OUTFILE_TYPE& f,
                int trc_start, int trc_end ) {

    auto itr = v.data();
    for( int traceno = trc_start; traceno <= trc_end; ++traceno )
        itr = f.put( traceno, itr );
}

template< typename T >
T normalization( const std::vector< matrix< T > >& wvlets ) {
    T norm = 0;

    for( auto & wvlet: wvlets ) {
        T m = wvlet.cwiseAbs().maxCoeff();
        norm = m > norm ? m : norm;
    }
    return norm;
}

template< typename T >
vector< T > reconstruct_data( Eigen::Ref< vector< T > > rel_AI,
                              const matrix< T >& A,
                              T norm,
                              std::size_t traces ) {
    const int tracelen = A.cols();

    Eigen::Map< matrix< T > > x( rel_AI.data(), tracelen, traces );
    // cppcheck-suppress unreadVariable
    x = A * x * norm;

    return rel_AI;
}

std::vector< std::pair< std::size_t, std::size_t > > segments( int numseg,
                                                               std::size_t fast,
                                                               std::size_t slow,
                                                               int overlap ) {

    /* Data is divided into <numseg> segments on fast, with an
     * overlap. This function returns (start, end) pairs containing the
     * first and last (inclusive) trace of the segments.
     */

    std::vector< std::pair< std::size_t, std::size_t > > sgmnts;

    for( int i = 0; i < numseg; ++i ) {
        std::size_t first = slow * std::round( double(i) / numseg * fast );
        std::size_t last = slow * ( std::round( double(i+1)/numseg * fast
                                      + overlap ) + 1 )
                                  - 1;
        last = std::min( last, fast*slow - 1 );
        sgmnts.push_back( { first, last } );
    }

    return sgmnts;
}

template<typename INFILE_TYPE, typename OUTFILE_TYPE, typename OPTIONS>
void compute_impedance_of_full_cube( std::vector< INFILE_TYPE >& files,
                                     std::vector< OUTFILE_TYPE >& relAI_files,
                                     std::vector< OUTFILE_TYPE >& dsyn_files,
                                     OPTIONS& opts ) {
    using T = float;

    if( opts.overlap < 0 ) opts.overlap = opts.max_iter;

    Progress::expected += opts.segments * ( opts.max_iter + 25 );

    std::vector< matrix< T > > wvlets = wavelets< T >( files,
                                                       opts.tv_wavelet,
                                                       opts.polarity );

    Progress::report( 5 );

    T norm = normalization( wvlets );
    const int vintages = files.size();

    std::vector< matrix< T > > A = forward_operators< T >( wvlets,
                                                           vintages,
                                                           norm );

    const bool xl_sorted = files.front().sorting() == segyio::sorting::xline();
    const std::size_t fast = xl_sorted ? files.front().crosslinecount()
                                       : files.front().inlinecount();
    const std::size_t slow = xl_sorted ? files.front().inlinecount()
                                       : files.front().crosslinecount();

    const std::size_t tracelen = files.front().samplecount();

    const auto sgments = segments( opts.segments,
                                   fast, slow,
                                   opts.overlap );

    for( const auto& segment : sgments ) {
        const std::size_t trc_start = segment.first;
        const std::size_t trc_end = segment.second;

        vector< T > relAI = compute_impedance< T >( files,
                                                    relAI_files,
                                                    A,
                                                    norm,
                                                    opts.max_iter,
                                                    opts.damping_3D,
                                                    opts.damping_4D,
                                                    opts.latsmooth_3D,
                                                    opts.latsmooth_4D,
                                                    trc_start, trc_end );

        const std::size_t traces = trc_end - trc_start + 1;
        const std::size_t cubesize = traces * tracelen;

        for( int i = 0; i < vintages; ++i ) {
            auto seg = relAI.segment( i * cubesize, cubesize );

            writefile( seg,
                       relAI_files[ i ],
                       trc_start, trc_end );

            seg = reconstruct_data< T >( seg, A[ i ], norm, traces );

            writefile( seg,
                       dsyn_files[ i ],
                       trc_start, trc_end );
        }
        Progress::report( 5 );
    }
}

}

namespace Eigen { namespace internal {

template< typename T, typename Reporter >
struct traits< Si4tiImpMatrix< T, Reporter > > :
    public Eigen::internal::traits< Eigen::SparseMatrix< T > >
{};

template< typename T, typename Rhs, typename Reporter >
struct generic_product_impl< Si4tiImpMatrix< T, Reporter >,
                             Rhs,
                             SparseShape,
                             DenseShape,
                             GemvProduct // GEMV stands for matrix-vector
                           >
     : generic_product_impl_base<
            Si4tiImpMatrix< T, Reporter >,
            Rhs,
            generic_product_impl< Si4tiImpMatrix< T, Reporter >, Rhs >
       >
{

    using Scalar = typename Product< Si4tiImpMatrix< T, Reporter >, Rhs >::Scalar;

    template< typename Dest >
    static void scaleAndAddTo( Dest& dst,
                               const Si4tiImpMatrix< T, Reporter >& lhs,
                               const Rhs& rhs,
                               const Scalar& alpha ) {

        const std::size_t tracelen = lhs.mat[0].cols();
        const std::size_t slow = lhs.slow;
        const std::size_t fast = lhs.fast;
        const std::size_t traces = slow * fast;
        const std::size_t cubesize = tracelen * traces;
        const std::size_t vintages = lhs.vintages;

        const T damping_4D = lhs.damping_4D;
        const T lat_smooth_3D = lhs.lat_smooth_3D;
        const T lat_smooth_4D = lhs.lat_smooth_4D;

        matrix< T > sm( tracelen, vintages );

        for( int v = 0; v < vintages; ++v ) {
            Eigen::Map< matrix< T > > x( (T*) rhs.data() + v*cubesize,
                                         tracelen, traces );
            Eigen::Map< matrix< T > > b( (T*) dst.data() + v*cubesize,
                                         tracelen, traces );
            b = lhs.mat[v] * x;
        }

        #pragma omp parallel for firstprivate( sm ) schedule( guided )
        for( std::size_t t = 0; t < traces; ++t ) {

            std::size_t j = t / slow;
            std::size_t k = t % slow;

            std::vector< std::size_t > iss;

            if( lhs.segmented and t < lhs.slow )
                iss = {
                    // C(j-1, k)
                    k == 0 ? t : t - 1,
                    // C(j+1, k)
                    k == slow - 1 ? t : t + 1,
                    // C(j, k+1)
                    j == fast - 1 ? t : t + slow,
                };
            else
                iss = {
                    // C(j-1, k)
                    k == 0 ? t : t - 1,
                    // C(j+1, k)
                    k == slow - 1 ? t : t + 1,
                    // C(j, k-1)
                    j == 0 ? t : t - slow,
                    // C(j, k+1)
                    j == fast - 1 ? t : t + slow,
                };

            for( int v = 0; v < vintages; ++v ) {
                const std::size_t offset = v * cubesize + t * tracelen;

                sm.col( v ) = rhs.segment( offset, tracelen );
                for( const auto is : iss ) {
                    const std::size_t neighbour_offset =
                        is * tracelen + v * cubesize;

                    sm.col( v ) -= 0.25
                                * rhs.segment( neighbour_offset, tracelen );
                }

                dst.segment( offset, tracelen ) += lat_smooth_3D * sm.col( v );
            }

            for( int v1 = 0; v1 < vintages; ++v1 ) {
            for( int v2 = 0; v2 < vintages; ++v2 ) {

                if( v1 != v2 ) {
                    const std::size_t v1_offset = t * tracelen + v1 * cubesize;
                    const std::size_t v2_offset = t * tracelen + v2 * cubesize;

                    auto lsmth_4D = lat_smooth_4D * (sm.col(v1) - sm.col(v2));
                    auto dmp_4D = damping_4D
                                * ( rhs.segment( v1_offset, tracelen )
                                  - rhs.segment( v2_offset, tracelen ) );

                    dst.segment( v1_offset, tracelen )
                        += lsmth_4D + dmp_4D;
                }
            }}
        }
    }
};

} }
#endif /* IMPEDANCE_HPP */
