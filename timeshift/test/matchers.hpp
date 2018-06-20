#include <Eigen/Core>

#include <catch/catch.hpp>

namespace {

template< typename T >
class ApproxRangeMatcher : public Catch::MatcherBase< Eigen::Matrix< T, 1, Eigen::Dynamic > > {
    public:
        explicit ApproxRangeMatcher( const vector< T >& xs ) :
            lhs( xs )
        {}

        virtual bool match( const Eigen::Matrix< T, 1, Eigen::Dynamic >& xs ) const override {
            if( xs.size() != lhs.size() ) return false;

            for( size_t i = 0; i < xs.size(); ++i )
                if( xs[ i ] != Approx(this->lhs[ i ]) ) return false;

            return true;
        }

        virtual std::string describe() const override {
            using str = Catch::StringMaker< Eigen::Matrix< T, 1, Eigen::Dynamic > >;
            return "~= " + str::convert( this->lhs );
        }

    private:
        Eigen::Matrix< T, 1, Eigen::Dynamic > lhs;
};

template< typename T >
ApproxRangeMatcher< T > ApproxRange( const Eigen::Matrix< T, 1, Eigen::Dynamic >& xs ) {
    return ApproxRangeMatcher< T >( xs );
}

}
