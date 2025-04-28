#include <cassert>
#include <utility>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "impedance.hpp"

namespace py = pybind11;

namespace si4ti {
namespace python {

// Wrapper around a NumPy array to ensure that has the same interface like the
// `input_file` and `output_file` types used in the impedance code. This allows
// us to reuse the impedance code for the Python interface without any major
// changes.
class Si4tiNumpyWrapper {
    py::array_t< float > data_;

    std::pair< std::size_t, std::size_t >
    to_fast_and_slow_index( const int tracenr ) const {
        assert( tracenr > -1 );
        assert( tracenr < this->fastindexcount() * this->slowindexcount() );
        const std::size_t slow_index = tracenr % this->slowindexcount();
        const std::size_t fast_index =
            ( tracenr - slow_index ) / this->slowindexcount();
        assert( fast_index < this->fastindexcount() );
        assert( slow_index < this->slowindexcount() );
        return { fast_index, slow_index };
    }

  public:
    explicit Si4tiNumpyWrapper( py::array_t< float >&& data ) : data_( data ) {}

    // The segyio documentation [1] states the following meaning/order
    // of indices for post-stack cubes is normalised:
    //
    //      If it is post-stack (only the one offset), the dimensions are
    //      normalised to (fast, slow, sample)
    //
    // [1]: https://segyio.readthedocs.io/en/latest/segyio.html#segyio.tools.cube
    int fastindexcount() const { return this->data_.shape( 0 ); }
    int slowindexcount() const { return this->data_.shape( 1 ); }
    int samplecount() const { return this->data_.shape( 2 ); }

    int tracecount() const { return this->fastindexcount() * this->slowindexcount(); }

    py::array_t< float > release_data() { return std::move( this->data_ ); }

    template< typename InputIt >
    InputIt* put( int tracenr, InputIt* in ) {
        auto r = this->data_.template mutable_unchecked< 3 >();
        const auto indices = this->to_fast_and_slow_index( tracenr );
        const auto fast_index = indices.first;
        const auto slow_index = indices.second;
        std::copy_n( in, this->samplecount(),
                     r.mutable_data( fast_index, slow_index, 0 ) );
        return in + this->samplecount();
    }

    template< typename OutputIt >
    OutputIt* get( int tracenr, OutputIt* out ) const {
        auto r = this->data_.template unchecked< 3 >();
        const auto indices = this->to_fast_and_slow_index( tracenr );
        const auto fast_index = indices.first;
        const auto slow_index = indices.second;
        return std::copy_n( r.data( fast_index, slow_index, 0 ), this->samplecount(),
                            out );
    }
};

std::pair< py::list, py::list > impedance( const py::list& input,
                                           const ImpedanceOptions& options ) {
    std::vector< Si4tiNumpyWrapper > input_files;

    std::vector< Si4tiNumpyWrapper > relAI_arrays;
    std::vector< Si4tiNumpyWrapper > dsyn_arrays;

    for ( py::handle item : input ) {
        if ( not py::isinstance< py::array >( item ) ) {
            throw std::runtime_error(
                "All items in the input list must be NumPy arrays." );
        }

        py::array_t< float > numpy_array = py::cast< py::array >( item );
        if ( numpy_array.strides( 2 ) != sizeof( float ) ) {
            throw std::runtime_error(
                "The NumPy arrays must be contiguous in trace direction." );
        }

        const py::ssize_t shape[3]{ numpy_array.shape( 0 ), numpy_array.shape( 1 ),
                                    numpy_array.shape( 2 ) };
        const py::ssize_t strides[3]{ numpy_array.strides( 0 ),
                                      numpy_array.strides( 1 ),
                                      numpy_array.strides( 2 ) };

        input_files.emplace_back( Si4tiNumpyWrapper( std::move( numpy_array ) ) );

        relAI_arrays.emplace_back(
            Si4tiNumpyWrapper( py::array_t< float >( shape, strides ) ) );
        dsyn_arrays.emplace_back(
            Si4tiNumpyWrapper( py::array_t< float >( shape, strides ) ) );
    }

    compute_impedance_of_full_cube( input_files, relAI_arrays, dsyn_arrays, options );

    auto to_python_list = []( std::vector< Si4tiNumpyWrapper >&& data ) {
        py::list tmp;
        for ( auto& d : data ) {
            tmp.append( d.release_data() );
        }
        return tmp;
    };

    return { to_python_list( std::move( relAI_arrays ) ),
             to_python_list( std::move( dsyn_arrays ) ) };
}

} /* namespace python */
} /* namespace si4ti */

PYBIND11_MODULE( _si4ti_python, m ) {
    m.def( "impedance", &si4ti::python::impedance,
           "Computes the impedance from provided cubes and parameters",
           py::arg( "input" ), py::arg( "options" ), py::return_value_policy::move );

    py::class_< ImpedanceOptions >( m, "ImpedanceOptions" )
        .def( py::init<>(),
              R"pbdoc(
            Impedance options, equivalent to the CLI options.
            )pbdoc" )
        .def_readwrite( "polarity", &ImpedanceOptions::polarity )
        .def_readwrite( "segments", &ImpedanceOptions::segments )
        .def_readwrite( "overlap", &ImpedanceOptions::overlap )
        .def_readwrite( "tv_wavelet", &ImpedanceOptions::tv_wavelet )
        .def_readwrite( "damping_3D", &ImpedanceOptions::damping_3D )
        .def_readwrite( "damping_4D", &ImpedanceOptions::damping_4D )
        .def_readwrite( "latsmooth_3D", &ImpedanceOptions::latsmooth_3D )
        .def_readwrite( "latsmooth_4D", &ImpedanceOptions::latsmooth_4D )
        .def_readwrite( "max_iter", &ImpedanceOptions::max_iter );
}
