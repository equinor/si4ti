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
using input_file = segyio::basic_volume< segyio::readonly >;
using output_file = segyio::basic_volume< segyio::trace_writer,
                                          segyio::write_always >;

namespace {

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

template< typename T >
vector< T > timeinvariant_wavelet( input_file& survey ) {

    const auto tracelen = survey.samplecount();

    vector< T > fwav( tracelen );
    fwav.setZero();

    vector< T > trace( tracelen );
    Eigen::FFT< T > fft;
    vector< std::complex< T > > spectrum;

    for( int trc = 0; trc < survey.tracecount(); ++trc ) {
        survey.get( trc, trace.data() );
        fft.fwd( spectrum, trace );
        fwav += spectrum.cwiseAbs() / survey.tracecount();
    }

    return fwav;
}

template< typename T >
matrix< T > timevarying_wavelet( input_file& survey ) {

    const auto tracelen = survey.samplecount();
    const int win_size = 101;
    const int win_step = 20;

    matrix< T > fwav = matrix< T >::Zero( tracelen, tracelen );

    vector< T > trace( tracelen );
    Eigen::FFT< T > fft;
    vector< std::complex< T > > spectrum( win_size );

    const int last_offset = ((tracelen-win_size)/win_step)*win_step + win_size/2;

    for( int trc = 0; trc < survey.tracecount(); ++trc ) {
        survey.get( trc, trace.data() );
        for( int offset = win_size/2; offset <= last_offset; offset += win_step ) {
            auto segment = trace.segment( offset-win_size/2, win_size );
            fft.fwd( spectrum, segment );
            fwav.col( offset ).head( win_size )
                += spectrum.cwiseAbs() / survey.tracecount();
        }
    }

    fwav.leftCols( win_size/2 ).colwise() = fwav.col( win_size/2 );
    fwav.rightCols( tracelen - last_offset ).colwise() = fwav.col( last_offset );

    for( int offset = win_size/2; offset < last_offset; offset += win_step ) {
        for( int i = 1; i < win_step; ++i ) {
            fwav.col( offset + i )
                = (1-T(i)/win_step)*fwav.col( offset )
                + (T(i)/win_step)*(fwav.col( offset + win_step ));
        }
    }

    for( int c = 0; c < tracelen; ++c ) {
        auto segment = fwav.col(c).head( win_size );
        spectrum = segment;
        fft.inv( segment, spectrum );
    }

    matrix< T > mh( win_size, tracelen );
    mh.colwise() = myhamn< T >( win_size );

    ifftshift< T >( fwav.topRows( win_size ) );
    fwav.topRows( win_size ) = mh.array() * fwav.topRows( win_size ).array();

    vector< std::complex< T > > spectrum2( tracelen );
    for( int c = 0; c < tracelen; ++c ) {
        auto col = fwav.col(c);
        fft.fwd( spectrum2, col );
        col = spectrum2.cwiseAbs();
    }

    fwav.array().rowwise() /=
           fwav.colwise().mean().array() / fwav.mean()
         + std::numeric_limits<T>::epsilon();

    return fwav;
}

template< typename T >
std::vector< matrix< T > > wavelets( std::vector< input_file >& vintages,
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

template< typename T >
solution_1D< T > solve_1D( std::vector< input_file >&  vintages,
                           const std::vector< matrix< T > >& L,
                           const std::vector< matrix< T > >& A,
                           T damping,
                           int trc_start, int trc_end ) {

    const int traces =  trc_end - trc_start + 1;
    const int tracelen = L[0].cols();
    const int cubesize = traces * tracelen;
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

        for( int t = trc_start; t <= trc_end; ++t ) {
            const int offset = v * cubesize + (t-trc_start) * tracelen;
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

template< typename T >
void add_boundary_inline( std::vector< output_file >& relAI_files,
                          vector< T >& b,
                          T norm,
                          T lat_smooth_3D, T lat_smooth_4D,
                          int trc_start, int trc_end ) {
    /*
     * Adds lateral 3D/4D smoothing and 4D damping contributions from the last
     * trace of the previous segment.
     */

    const int nvints = relAI_files.size();
    const int xlines = relAI_files.front().crosslinecount();
    const int traces = trc_end - trc_start + 1;
    const int tracelen = relAI_files.front().samplecount();
    const int cubesize = tracelen * traces;

    vector< T > trace( tracelen );
    vector< T > trace2( tracelen );

    for( int v = 0; v < nvints; ++v ) {
        for( int t = (trc_start - xlines); t < trc_start; ++t ) {
            const int offset = v * cubesize + (t-(trc_start-xlines)) * tracelen;
            relAI_files[v].get( t, trace.data() );
            b.segment( offset, tracelen ) += norm * (lat_smooth_3D/4) * trace;

            for( int v2 = 0; v2 < nvints; ++v2 ) {
                if( v != v2 ) {
                    relAI_files[v2].get( t, trace2.data() );
                    b.segment( offset, tracelen ) +=
                        norm * (lat_smooth_4D/4) * ( trace - trace2 );
                }
            }
        }
    }
}

struct QuietReporter {
    static void report(){};
};

template< typename T, typename Reporter = QuietReporter >
struct SimpliImpMatrix : public Eigen::EigenBase< SimpliImpMatrix< T, Reporter > > {
    using Scalar = T;
    using RealScalar = T;
    using StorageIndex = std::ptrdiff_t;
    using Index = typename Eigen::EigenBase< SimpliImpMatrix >::Index;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    SimpliImpMatrix( std::vector< matrix< T > > m,
                     int vints,
                     int inlines,
                     int crosslines,
                     T damping_4D,
                     T lat_smooth_3D,
                     T lat_smooth_4D,
                     bool sgmented ) :
        mat( std::move( m ) ),
        vintages( vints ),
        ilines( inlines ),
        xlines( crosslines ),
        damping_4D( damping_4D ),
        lat_smooth_3D( lat_smooth_3D ),
        lat_smooth_4D( lat_smooth_4D ),
        segmented( sgmented )
    {}

    Index rows() const {
        return this->mat[0].rows() * vintages * xlines * ilines;
    }

    Index cols() const {
        return this->mat[0].rows() * vintages * xlines * ilines;
    }

    template< typename Rhs >
    Eigen::Product< SimpliImpMatrix, Rhs, Eigen::AliasFreeProduct >
    operator*( const Eigen::MatrixBase< Rhs >& x ) const {
        Reporter::report();
        return { *this, x.derived() };
    }

    std::vector< matrix< T > > mat;
    int vintages;
    int ilines, xlines;
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

template< typename T >
vector< T > compute_impedance( std::vector< input_file >& vintages,
                               std::vector< output_file >& relAI_files,
                               std::vector< matrix< T > >& A,
                               T norm,
                               int max_iter,
                               T damping_3D,
                               T damping_4D,
                               T lat_smooth_3D,
                               T lat_smooth_4D,
                               int trc_start,
                               int trc_end ) {

    const int xlines = vintages.front().crosslinecount();
    const int ilines = (trc_end - trc_start + 1) / xlines;
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

    SimpliImpMatrix< T, Progress > rbd_L( std::move( L ),
                                          nvints,
                                          ilines,
                                          xlines,
                                          damping_4D,
                                          lat_smooth_3D,
                                          lat_smooth_4D,
                                          segmented );

    conjugate_gradient( rbd_L, sol.rj, sol.b, max_iter );

    return sol.rj / norm;
}

template< typename Vector >
void writefile( const Vector& v,
                output_file& f,
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
                              int traces ) {
    const int tracelen = A.cols();

    Eigen::Map< matrix< T > > x( rel_AI.data(), tracelen, traces );
    x = A * x * norm;

    return rel_AI;
}

std::vector< std::pair< int, int > > segments( int numseg,
                                               int ilines,
                                               int xlines,
                                               int overlap ) {

    /* Data is divided into <numseg> segments on inlines, with an
     * overlap. This function returns (start, end) pairs containing the
     * first and last (inclusive) trace of the segments.
     */

    std::vector< std::pair<  int, int > > sgmnts;

    for( int i = 0; i < numseg; ++i ) {
        int first = xlines * std::round( double(i) / numseg * ilines );
        int last = xlines * ( std::round( double(i+1)/numseg * ilines
                              + overlap ) + 1 )
                          - 1;
        last = std::min( last, ilines*xlines - 1 );
        sgmnts.push_back( { first, last } );
    }

    return sgmnts;
}

}

namespace Eigen { namespace internal {

template<>
template< typename T, typename Reporter >
struct traits< SimpliImpMatrix< T, Reporter > > :
    public Eigen::internal::traits< Eigen::SparseMatrix< T > >
{};

template< typename T, typename Rhs, typename Reporter >
struct generic_product_impl< SimpliImpMatrix< T, Reporter >,
                             Rhs,
                             SparseShape,
                             DenseShape,
                             GemvProduct // GEMV stands for matrix-vector
                           >
     : generic_product_impl_base<
            SimpliImpMatrix< T, Reporter >,
            Rhs,
            generic_product_impl< SimpliImpMatrix< T, Reporter >, Rhs >
       >
{

    using Scalar = typename Product< SimpliImpMatrix< T, Reporter >, Rhs >::Scalar;

    template< typename Dest >
    static void scaleAndAddTo( Dest& dst,
                               const SimpliImpMatrix< T, Reporter >& lhs,
                               const Rhs& rhs,
                               const Scalar& alpha ) {

        const int tracelen = lhs.mat[0].cols();
        const int xlines = lhs.xlines;
        const int ilines = lhs.ilines;
        const int traces = xlines * ilines;
        const int cubesize = tracelen * traces;
        const int vintages = lhs.vintages;

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
        for( int t = 0; t < traces; ++t ) {

            int j = t / xlines;
            int k = t % xlines;

            std::vector< std::ptrdiff_t > iss;

            if( lhs.segmented and t < lhs.xlines )
                iss = {
                    // C(j-1, k)
                    k == 0 ? t : t - 1,
                    // C(j+1, k)
                    k == xlines - 1 ? t : t + 1,
                    // C(j, k+1)
                    j == ilines - 1 ? t : t + xlines,
                };
            else
                iss = {
                    // C(j-1, k)
                    k == 0 ? t : t - 1,
                    // C(j+1, k)
                    k == xlines - 1 ? t : t + 1,
                    // C(j, k-1)
                    j == 0 ? t : t - xlines,
                    // C(j, k+1)
                    j == ilines - 1 ? t : t + xlines,
                };

            for( int v = 0; v < vintages; ++v ) {
                const int offset = v * cubesize + t * tracelen;

                sm.col( v ) = rhs.segment( offset, tracelen );
                for( const auto is : iss ) {
                    const int neighbour_offset = is * tracelen + v * cubesize;
                    sm.col( v ) -= 0.25
                                * rhs.segment( neighbour_offset, tracelen );
                }

                dst.segment( offset, tracelen ) += lat_smooth_3D * sm.col( v );
            }

            for( int v1 = 0; v1 < vintages; ++v1 ) {
            for( int v2 = 0; v2 < vintages; ++v2 ) {

                if( v1 != v2 ) {
                    const int v1_offset = t * tracelen + v1 * cubesize;
                    const int v2_offset = t * tracelen + v2 * cubesize;

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
