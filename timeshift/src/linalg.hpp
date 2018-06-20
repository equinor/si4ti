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
    static void upper_triangle( Dest& dst,
                                const BlockBandedMatrix< T >& lhs,
                                const Rhs& rhs,
                                const Scalar& alpha ) {

        const auto traces       = lhs.ilines * lhs.xlines;
        const auto localsize    = lhs.Bnn.rows();
        const auto timeshifts   = lhs.vintages - 1;
        const auto diagonals    = lhs.diagonals;
        const auto vintpairsize = lhs.mat.rows() / timeshifts;

        # pragma omp parallel for schedule(guided)
        for( int trace = 0; trace < traces; ++trace ) {

        for( int mvrow = 0; mvrow < timeshifts; ++mvrow ) {
        for( int mvcol = 0; mvcol < timeshifts; ++mvcol ) {
        for( int diag = 0; diag < diagonals; ++diag ) {
            const auto lhs_col   = mvcol * diagonals + diag;
            const auto lhs_start = mvrow * vintpairsize + trace * localsize;
            const auto dst_start = lhs_start;
            const auto rhs_start = diag
                                 + mvcol * vintpairsize
                                 + trace * localsize;
            const auto len       = localsize - diag;

            dst.segment( dst_start, len ).array()
                += alpha * lhs.mat
                              .col( lhs_col )
                              .segment( lhs_start, len )
                              .array()
                         * rhs.segment( rhs_start, len )
                              .array()
                         ;

        }}}

        }
    }

    template< typename Dest >
    static void lower_triangle( Dest& dst,
                                const BlockBandedMatrix< T >& lhs,
                                const Rhs& rhs,
                                const Scalar& alpha ) {

        const auto traces       = lhs.ilines * lhs.xlines;
        const auto localsize    = lhs.Bnn.rows();
        const auto timeshifts   = lhs.vintages - 1;
        const auto diagonals    = lhs.diagonals;
        const auto vintpairsize = lhs.mat.rows() / timeshifts;

        # pragma omp parallel for schedule(guided)
        for( int trace = 0; trace < traces; ++trace ) {

        for( int mvrow = 0; mvrow < timeshifts; ++mvrow ) {
        for( int mvcol = 0; mvcol < timeshifts; ++mvcol ) {
        // NOTE: diag starts at 1
        for( int diag = 1; diag < diagonals; ++diag ) {

            const auto lhs_col   = mvcol * diagonals + diag;
            const auto lhs_start = mvrow * vintpairsize + trace * localsize;
            const auto dst_start = lhs_start;
            const auto rhs_start = diag
                                 + mvcol * vintpairsize
                                 + trace * localsize;
            const auto len       = localsize - diag;

            dst.segment( rhs_start, len ).array()
                += alpha * lhs.mat
                            .col( lhs_col )
                            .segment( lhs_start, len )
                            .array()
                        * rhs.segment( dst_start, len )
                            .array()
                        ;
        }}}

        }
    }

    template< typename Dest >
    static void apply_smoothing( Dest& dst,
                                 const BlockBandedMatrix< T >& lhs,
                                 const Rhs& rhs ) {

        const auto ilines     = lhs.ilines;
        const auto xlines     = lhs.xlines;
        const auto traces     = ilines * xlines;
        const auto localsize  = lhs.Bnn.rows();
        const auto timeshifts = lhs.vintages - 1;

        # pragma omp parallel for schedule(guided)
        for( int trace = 0; trace < traces; ++trace ) {

        const auto vintsize = localsize * traces;
        const auto& comb    = lhs.comb;

        for( int mvrow = 0; mvrow < timeshifts; ++mvrow ) {
        for( int mvcol = 0; mvcol < timeshifts; ++mvcol ) {

        int j = trace / xlines;
        int k = trace % xlines;
        int i = trace;
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

            const auto col = (mvrow * vintsize) + i * localsize;
            for( const auto is : iss ) {
                const auto row = (mvcol * vintsize) + is * localsize;
                const auto x = rhs.segment( row, localsize );
                const auto smoothing = comb(mvrow, mvcol) * (lhs.Bnn * x);
                dst.segment(col, localsize) -= smoothing;
            }

        }
        }
    }
    }

    template< typename Dest >
    static void scaleAndAddTo( Dest& dst,
                               const BlockBandedMatrix< T >& lhs,
                               const Rhs& rhs,
                               const Scalar& alpha ) {

        // TODO: draw segment -> system mapping
        /*
         * the different parts of this computation is split into functions for
         * clarity, and to provide natural barriers, because the computation of
         * upper overlaps with lower and nearest neighbours computations.
         *
         * TODO: try splitting on worker-pool and tasks, because race
         * conditions only apply when two traces (same number) are computed at
         * the same time
         */
        upper_triangle( dst, lhs, rhs, alpha );
        lower_triangle( dst, lhs, rhs, alpha );
        apply_smoothing( dst, lhs, rhs );
    }
};

} }
