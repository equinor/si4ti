#include <algorithm>
#include <vector>
#include <iostream>

#include <Python.h>
#include <bytesobject.h>

#include <sml/bspline.h>

namespace {

struct buffer_guard {
    buffer_guard( Py_buffer& b ) : p( &b ) {}
    ~buffer_guard() { if( this->p ) PyBuffer_Release( this->p ); }

    Py_buffer* p = nullptr;
};

PyObject* bspline( PyObject*, PyObject* args ) {
    Py_buffer bufknots;
    Py_buffer output;
    int samples;
    int order;

    if( !PyArg_ParseTuple( args, "s*s*ii",
                                    &bufknots,
                                    &output,
                                    &samples,
                                    &order ) )
        return nullptr;


    buffer_guard g1( bufknots );
    buffer_guard g2( output );

    int knotlen = bufknots.len / sizeof( float );
    auto knots = (float*)bufknots.buf;
    auto out = (float*)output.buf;

    constexpr auto layout = SML_LAYOUT_ROWMAJ;
    sml_bspline_matrixf( samples, knots, knotlen, order, layout, out );

    return Py_BuildValue( "" );
}

PyMethodDef methods[] = {
    { "bspline", (PyCFunction) bspline, METH_VARARGS, "B-spline as matrix." },
    { nullptr }
};

}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_bspline( void ) {

    static struct PyModuleDef bspline_module = {
            PyModuleDef_HEAD_INIT,
            "bspline",   /* name of module */
            NULL, 
            -1,  
            methods,
    };
    return PyModule_Create( &bspline_module );
}

#else

PyMODINIT_FUNC initbspline( void ) {
    (void)Py_InitModule( "bspline", methods );
}

#endif // PY3
