#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <utility>

//#include "impedance.hpp"

template<typename INFILE_TYPE, typename OUTFILE_TYPE, typename OPTIONS>
void compute_impedance_of_full_cube( std::vector< INFILE_TYPE >& files,
                                     std::vector< OUTFILE_TYPE >& relAI_files,
                                     std::vector< OUTFILE_TYPE >& dsyn_files,
                                     OPTIONS& opts );



int add(int i, int j) { return i + j; }

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


std::pair<py::list, py::list> impedance(
    const py::list& input,
    const ImpedanceOptions& options
) {

    // Check input dimensions.
    //if (input1.ndim() != 1 || input2.ndim() != 1)
    //    throw std::runtime_error("Input should be 1-D NumPy arrays.");

    py::list output1;
    py::list output2;
    for (py::handle item: input) {
        if (!py::isinstance<py::array>(item)) {
            throw std::runtime_error("All items in the input list must be NumPy arrays.");
        }

        py::array_t<float, py::array::c_style> arr = py::cast<py::array>(item);

        //if (arr.ndim() != 3)
        //    throw std::runtime_error("Input should be 3D NumPy arrays.");

        //auto buf = arr.request();
        auto r = arr.unchecked<3>();
        auto result1 = std::vector<float>(r.size());

        // Example processing: scale and cast the elements of the input arrays.

        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            for (py::ssize_t j = 0; j < r.shape(1); j++) {
                for (py::ssize_t k = 0; k < r.shape(2); k++) {
                    //sum += r(i, j, k);
                    result1[i * r.shape(1) * r.shape(2) + j * r.shape(2) + k] = static_cast<float>(2.0 * r(i, j, k));
                }
            }
        }

//        for (ssize_t i = 0; i < buf.size; i++)
//            result1[i] = static_cast<float>(2.0 * static_cast<float*>(buf.ptr)[i]);

        // Create NumPy arrays from the std::vector results.

        {
            auto tmp = py::array_t<float>(result1.size(), result1.data())
                .reshape({r.shape(0), r.shape(1), r.shape(2)});
;
            output1.append(tmp.reshape({r.shape(0), r.shape(1), r.shape(2)}));
            //output1.append( tmp );
        }
        {
            auto tmp = py::array_t<float>(result1.size(), result1.data())
                .reshape({r.shape(0), r.shape(1), r.shape(2)});
            output2.append( tmp );
        }
    }

    return {output1, output2};
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
      Pybind11 example plugin
      -----------------------
      .. currentmodule:: python_example
      .. autosummary::
         :toctree: _generate
         add
         subtract
  )pbdoc";

  m.def("add", &add, R"pbdoc(
      Add two numbers
      Some other explanation about the add function.
  )pbdoc");

  m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
      Subtract two numbers
      Some other explanation about the subtract function.
  )pbdoc");

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
        .def_readwrite("polarity", &ImpedanceOptions::polarity, "asdf")
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
