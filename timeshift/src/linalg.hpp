#include <Eigen/Core>
#include <Eigen/Sparse>

#include <omp.h>

namespace {

template< typename T >
using vector = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
template< typename T >
using matrix = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;
template< typename T >
using sparse = Eigen::SparseMatrix< T >;

template< typename T >
struct BlockBandedMatrix : public Eigen::EigenBase< BlockBandedMatrix< T > > {
    using Scalar = T;
    using RealScalar = T;
    using StorageIndex = std::ptrdiff_t;
    using Index = typename Eigen::EigenBase< BlockBandedMatrix >::Index;

    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    BlockBandedMatrix( matrix< T > m,
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
    Eigen::Product< BlockBandedMatrix, Rhs, Eigen::AliasFreeProduct >
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

}

namespace Eigen { namespace internal {

template<>
template< typename T >
struct traits< BlockBandedMatrix< T > > :
    public Eigen::internal::traits< Eigen::SparseMatrix< T > >
{};

template< typename T, typename Rhs >
struct generic_product_impl< BlockBandedMatrix< T >,
                             Rhs,
                             SparseShape,
                             DenseShape,
                             GemvProduct // GEMV stands for matrix-vector
                           >
     : generic_product_impl_base<
            BlockBandedMatrix< T >,
            Rhs,
            generic_product_impl< BlockBandedMatrix< T >, Rhs >
       >
{
    using Scalar = typename Product< BlockBandedMatrix< T >, Rhs >::Scalar;
    template< typename Dest >
    static void scaleAndAddTo( Dest& dst,
                               const BlockBandedMatrix< T >& lhs,
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
