#include <cassert>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "impedance.hpp"


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

class Si4tiNumpyWrapper {
    py::array_t<float, py::array::c_style> data_;

    std::pair<std::size_t, std::size_t> to_inline_crossline_nr(int tracenr) const {
        assert(tracenr < this->inlinecount() * this->crosslinecount());
        const std::size_t inlinenr = tracenr % this->inlinecount();
        assert(tracenr >= inlinenr);
        const std::size_t crosslinenr = (tracenr - inlinenr) / this->inlinecount();
        assert(inlinenr < this->inlinecount());
        assert(crosslinenr < this->crosslinecount());
        return {inlinenr, crosslinenr};
    }

public:
    explicit Si4tiNumpyWrapper(py::array_t<float, py::array::c_style> data)
        : data_(data)
    {
    }

    const py::array_t<float, py::array::c_style>& data() const {
        return this->data_;
    }
    // For NumPy arrays the notion of inline or crossline sorted does not exist
    // the same way as for SEG-Y files. The first index, i.e., is the slowest
    // and the last index is the fastest index. In this sense, NumPy arrays
    // should always appear as inline sorted.
    static constexpr bool xlinesorted() noexcept(true) {
        return true;
    }

    int inlinecount() const {
        return this->data_.shape(0);
    }

    int crosslinecount() const {
        return this->data_.shape(1);
    }

    int tracecount() const {
        return this->inlinecount() * this->crosslinecount();
    }

    int samplecount() const {
        return this->data_.shape(2);
    }

    py::array_t<float, py::array::c_style>&& data() {
        return std::move(this->data_);
    }


    template<typename InputIt>
    InputIt* put(int tracenr, InputIt* in) {
        auto r = this->data_.template mutable_unchecked<3>();
        const auto numbers = this->to_inline_crossline_nr(tracenr);
        const auto inlinenr = numbers.first;
        const auto crosslinenr = numbers.second;
        std::copy_n(in, this->samplecount(), r.mutable_data(inlinenr, crosslinenr, 0));
        return in + this->samplecount();
    }

    template<typename OutputIt>
    OutputIt* get(int tracenr, OutputIt* out) const {
        auto r = this->data_.template unchecked<3>();
        const auto numbers = this->to_inline_crossline_nr(tracenr);
        const auto inlinenr = numbers.first;
        const auto crosslinenr = numbers.second;
        return std::copy_n(r.data(inlinenr, crosslinenr, 0), this->samplecount(), out);
    }
};


py::list to_python_list(std::vector<Si4tiNumpyWrapper>&& data) {
    py::list tmp;
    for (const auto& d: data) {
        tmp.append(std::move(d.data()));
    }
    return tmp;
}

std::pair<py::list, py::list> impedance(
    const py::list& input,
    ImpedanceOptions options
) {
    std::vector<Si4tiNumpyWrapper> input_files;

    std::vector<Si4tiNumpyWrapper> relAI_arrays;
    std::vector<Si4tiNumpyWrapper> dsyn_arrays;

    for (py::handle item: input) {
        if (!py::isinstance<py::array>(item)) {
            throw std::runtime_error("All items in the input list must be NumPy arrays.");
        }

        py::array_t<float, py::array::c_style> numpy_array = py::cast<py::array>(item);
        input_files.push_back(Si4tiNumpyWrapper(numpy_array));

        const py::ssize_t shape[3]{numpy_array.shape(0), numpy_array.shape(1), numpy_array.shape(2)};
        const py::ssize_t strides[3]{numpy_array.strides(0), numpy_array.strides(1), numpy_array.strides(2)};

        relAI_arrays.push_back(Si4tiNumpyWrapper(std::move(py::array_t<float, py::array::c_style>(shape, strides))));
        dsyn_arrays.push_back(Si4tiNumpyWrapper(std::move(py::array_t<float, py::array::c_style>(shape, strides))));
    }

    compute_impedance_of_full_cube(input_files, relAI_arrays, dsyn_arrays, options);

    return {
        to_python_list(std::move(relAI_arrays)),
        to_python_list(std::move(dsyn_arrays))
    };
}

PYBIND11_MODULE(_si4ti_python, m) {
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
        py::return_value_policy::move,
        R"pbdoc(
            TBD: Write documentation of impedance functions
    )pbdoc");

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
