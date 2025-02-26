#include <cassert>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
//#include <segyio/segyio.hpp>

#include "impedance.hpp"

//template<typename INFILE_TYPE, typename OUTFILE_TYPE, typename OPTIONS>
//void compute_impedance_of_full_cube( std::vector< INFILE_TYPE >& files,
//                                     std::vector< OUTFILE_TYPE >& relAI_files,
//                                     std::vector< OUTFILE_TYPE >& dsyn_files,
//                                     OPTIONS& opts );



namespace py = pybind11;

struct ImpedanceOptions {
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

// TODO: Investigate if we can avoid copies
class NumpyIOFile {
    // The actual memory is being held/allocated somewhere else
    py::array_t<float> data_;

public:
    NumpyIOFile(py::array_t<float> data)
        : data_(data)
    {
    }

    //// Takes ownerwhip of the data
    //NumpyIOFile(py::array_t<float>&& data)
    //: data_(std::move(data))
    //{
    //}

    const py::array_t<float>& data() const {
        return this->data_;
    }
    //segyio::sorting sorting() const {
    //    // TODO: We arbitrarily set the sorting to some value here. This has to
    //    // be corrected!
    //    return segyio::sorting::iline();
    //}
    static constexpr bool xlinesorted() const noexcept(true) {
        return false;
    }

    int inlinecount() const {
        return (not this->xlinesorted()) ? this->data_.shape(0) : this->data_.shape(1);
        //if (this->sorting() == segyio::sorting::iline()) {
        //    return this->data_.shape(0);
        //} else {
        //    return this->data_.shape(1);
        //}
    }

    int crosslinecount() const {
        return (not this->xlinesorted()) ? this->data_.shape(1) : this->data_.shape(0);
        //if (this->sorting() == segyio::sorting::iline()) {
        //    return this->data_.shape(1);
        //} else {
        //    return this->data_.shape(0);
        //}
    }

    int tracecount() const {
        return this->inlinecount() * this->crosslinecount();
    }

    int samplecount() const {
        return this->data_.shape(2);
    }



    //OutputIt trace_reader< Derived >::get( int i, OutputIt out ) noexcept(false) {
    template<typename InputIt>
    InputIt* put(int tracenr, InputIt* in) {
        //auto r = this->data_.unchecked<3>();
        //const auto offset = this->trace_offset(tracenr);
        //std::copy_n(in, this->samplecount(), &r[offset]);
        //return in + this->samplecount();
        auto r = this->data_.mutable_unchecked<3>();
        const auto [inlinenr, crosslinenr] = this->to_inline_crossline_nr(tracenr);
        std::copy_n(in, this->samplecount(), r.mutable_data(inlinenr, crosslinenr, 0));
        return in + this->samplecount();
    }

    template<typename OutputIt>
    OutputIt* get(int tracenr, OutputIt* out) const {
        // TODO: This needs to do some copying of data.
        //auto r = this->data_.unchecked<3>();
        //const auto offset = this->trace_offset(tracenr);
        //return std::copy_n(&r[offset], this->samplecount(), out);

        auto r = this->data_.unchecked<3>();
        const auto [inlinenr, crosslinenr] = this->to_inline_crossline_nr(tracenr);
        return std::copy_n(r.data(inlinenr, crosslinenr, 0), this->samplecount(), out);
    }

    //segyio::sorting sorting() const noexcept(true) {
    //    // We artificially set the sorting to some value
    //    return segyio::sorting::xline();
    //}
    //int inlinecount()         const noexcept(true) {
    //    return this->data_.shape(0);
    //}
    //int crosslinecount()      const noexcept(true) {
    //    return this->data_.shape(1);
    //}

    //int offsetcount()         const noexcept(true) {
    //    return this->data_.shape(2);
    //}
    //OutputIt trace_reader< Derived >::get( int i, OutputIt out ) noexcept(false) {

private:
    std::pair<std::size_t, std::size_t> to_inline_crossline_nr(int tracenr) const {
        assert(tracenr < this->inlinecount() * this->crosslinecount());
        if (not this->xlinesorted()) {
            const std::size_t crosslinenr = tracenr % this->crosslinecount();
            const std::size_t inlinenr = (tracenr - crosslinenr) / this->crosslinecount();
            return std::pair{inlinenr, crosslinenr};
        } else {
            // crossline sorted
            const std::size_t inlinenr = tracenr % this->inlinecount();
            const std::size_t crosslinenr = (tracenr - inlinenr) / this->inlinecount();
            return std::pair{inlinenr, crosslinenr};
        }

        //return static_cast<std::size_t>(tracenr) * this->samplecount();
    }

};



std::pair<py::list, py::list> impedance(
    const py::list& input,
    ImpedanceOptions options
) {

    // Check input dimensions.
    //if (input1.ndim() != 1 || input2.ndim() != 1)
    //    throw std::runtime_error("Input should be 1-D NumPy arrays.");

    //py::list output1;
    //py::list output2;

    std::vector<NumpyIOFile> input_files;

    std::vector<NumpyIOFile> output_1;
    std::vector<NumpyIOFile> output_2;

    for (py::handle item: input) {
        if (!py::isinstance<py::array>(item)) {
            throw std::runtime_error("All items in the input list must be NumPy arrays.");
        }

        py::array_t<float, py::array::c_style> numpy_array = py::cast<py::array>(item);
        input_files.push_back(NumpyIOFile(numpy_array));

        //constexpr size_t elsize = sizeof(double);
        //size_t shape[3]{100, 1000, 1000};
        //size_t strides[3]{1000 * 1000 * elsize, 1000 * elsize, elsize};

        //auto out_1 = py::array_t<float>(numpy_array.shape(), numpy_array.strides());
        const py::ssize_t shape[3]{numpy_array.shape(0), numpy_array.shape(1), numpy_array.shape(2)};
        const py::ssize_t strides[3]{numpy_array.strides(0), numpy_array.strides(1), numpy_array.strides(2)};
        auto out_1 = py::array_t<float>(shape, strides);
        auto out_2 = py::array_t<float>(shape, strides);

        output_1.push_back(out_1);
        output_2.push_back(out_2);
    }

    // Do stuff
    compute_impedance_of_full_cube(input_files, output_1, output_2, options);

    // Copy data back
    py::list output1;
    py::list output2;

    for (const auto& out: output_1) {
        output1.append(out.data());
    }

    for (const auto& out: output_2) {
        output2.append(out.data());
    }

    //for (py::handle item: input) {
    //    if (!py::isinstance<py::array>(item)) {
    //        throw std::runtime_error("All items in the input list must be NumPy arrays.");
    //    }

    //    py::array_t<float, py::array::c_style> arr = py::cast<py::array>(item);

    //    //if (arr.ndim() != 3)
    //    //    throw std::runtime_error("Input should be 3D NumPy arrays.");

    //    //auto buf = arr.request();
    //    auto r = arr.unchecked<3>();
    //    auto result1 = std::vector<float>(r.size());

    //    // Example processing: scale and cast the elements of the input arrays.

    //    for (py::ssize_t i = 0; i < r.shape(0); i++) {
    //        for (py::ssize_t j = 0; j < r.shape(1); j++) {
    //            for (py::ssize_t k = 0; k < r.shape(2); k++) {
    //                //sum += r(i, j, k);
    //                result1[i * r.shape(1) * r.shape(2) + j * r.shape(2) + k] = static_cast<float>(2.0 * r(i, j, k));
    //            }
    //        }
    //    }

//  //      for (ssize_t i = 0; i < buf.size; i++)
//  //          result1[i] = static_cast<float>(2.0 * static_cast<float*>(buf.ptr)[i]);

    //    // Create NumPy arrays from the std::vector results.

    //    {
    //        auto tmp = py::array_t<float>(result1.size(), result1.data())
    //            .reshape({r.shape(0), r.shape(1), r.shape(2)});
;
    //        output1.append(tmp.reshape({r.shape(0), r.shape(1), r.shape(2)}));
    //        //output1.append( tmp );
    //    }
    //    {
    //        auto tmp = py::array_t<float>(result1.size(), result1.data())
    //            .reshape({r.shape(0), r.shape(1), r.shape(2)});
    //        output2.append( tmp );
    //    }
    //}


    return {output1, output2};
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
      Pybind11 example plugin
      -----------------------
      .. currentmodule:: python_example
      .. autosummary::
         :toctree: _generate
  )pbdoc";

    m.def(
        "impedance",
        &impedance,
        "A function that processes two NumPy arrays and returns two NumPy arrays.",
        py::arg("input"),
        py::arg("options"),
        R"pbdoc(
            TBD: Write documentation of impedance functions
    )pbdoc");


    //py::class_<ImpedanceOptions>(m, "ImpedanceOptions")
    //    .def(py::init<>())
    //    .def_readwrite("polarity", &ImpedanceOptions::polarity, "asdf")
    //    .def_readwrite("segments", &ImpedanceOptions::segments)
    //    .def_readwrite("overlap", &ImpedanceOptions::overlap)
    //    .def_readwrite("tv_wavelet", &ImpedanceOptions::tv_wavelet)
    //    .def_readwrite("damping_3D", &ImpedanceOptions::damping_3D)
    //    .def_readwrite("damping_4D", &ImpedanceOptions::damping_4D)
    //    .def_readwrite("latsmooth_3D", &ImpedanceOptions::latsmooth_3D)
    //    .def_readwrite("latsmooth_4D", &ImpedanceOptions::latsmooth_4D)
    //    .def_readwrite("max_iter", &ImpedanceOptions::max_iter);

    py::class_<ImpedanceOptions>(m, "ImpedanceOptions")
        .def(py::init<>(),
            R"pbdoc(
                Impedance options
            )pbdoc")
        .def_readwrite("polarity", &ImpedanceOptions::polarity)
        .def_readwrite("segments", &ImpedanceOptions::segments)
        .def_readwrite("overlap", &ImpedanceOptions::overlap)
        .def_readwrite("tv_wavelet", &ImpedanceOptions::tv_wavelet)
        .def_readwrite("damping_3D", &ImpedanceOptions::damping_3D)
        .def_readwrite("damping_4D", &ImpedanceOptions::damping_4D)
        .def_readwrite("latsmooth_3D", &ImpedanceOptions::latsmooth_3D)
        .def_readwrite("latsmooth_4D", &ImpedanceOptions::latsmooth_4D)
        .def_readwrite("max_iter", &ImpedanceOptions::max_iter);


  m.attr("__version__") = "dev";
}
