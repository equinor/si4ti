#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <Python.h>
#include <bytesobject.h>

#include <Eigen/Core>

namespace {

/* IMPLEMENTATIONS */

/*
 * Analyse the B-spline with De Boor's algorithm as an n * basis-functions
 * matrix.
 *
 * The samples 't' is the linear space [0,1] of interval dx = 1/(n-1), padded on
 * both sides according to the order of the spline curve, resulting in the
 * following intervals in start/step/stop notation:
 *
 * | t < 0:       [-dx * padlen : dx : 0              )
 * | t <= 0 <= 1: [0            : dx : 1              ]
 * | t > 1:       [1 + dx       : dx : padlen * dx + 1]
 *
 * The knots knotv are padded similarly, depending on the order of the curve,
 * so that:
 * | pre:   [-dx * order : dx : -dx * (order - padlen)]
 * | knots: knotv
 * | post:  [1 + dx      : dx : 1 + padlen * dx       ]
 *
 * Please note the two padlens are different for samples and knots.
 */
template< typename Numeric, int... Options >
Eigen::Matrix< Numeric, Eigen::Dynamic, Eigen::Dynamic, Options... >
bspline_matrix( int n, const Numeric* knotv, int knotlen, int order ) {
    using array = Eigen::Array< Numeric, Eigen::Dynamic, 1 >;
    using matrix = decltype( bspline_matrix( n, knotv, knotlen, order ) );

    auto samples = []( int n, int order ) {
        const Numeric interval = 1.0 / (n - 1.0);
        const auto padlen = 10 * order;
        array t( 2 * padlen + n );

        for( int i = 0; i < padlen; ++i )
            t[i] = -interval * (padlen - i);

        for( int i = 0; i < n; ++i )
            t[padlen + i] = i * interval;

        for( int i = 1; i <= padlen; ++i )
            t[padlen + n + i-1] = 1.0 + i*interval;

        return t;
    };

    auto padknots = []( int n, const Numeric* knotv, int knotlen, int order ) {
        const Numeric interval = 10.0 / (n - 1.0);

        const auto padlen = order + 1;
        std::vector< Numeric > knots( knotlen + 2*padlen );

        for( int i = 0; i <= padlen; ++i )
            knots[i] = -interval * (order - i);

        const auto scale = std::accumulate(
            knotv,
            knotv + knotlen,
            false,
            []( bool acc, Numeric x ) { return acc or x <= 0 or 1 <= x; }
        );

        const auto denom = scale ? n + 1 : 1;

        std::transform( knotv,
                        knotv + knotlen,
                        knots.begin() + padlen,
                        [=]( Numeric x ) { return x / denom; }
        );

        for( int i = 0; i < padlen; ++i )
            knots[padlen + knotlen + i] = 1.0 + i*interval;

        return knots;
    };

    const auto t = samples( n, order );
    const auto knots = padknots( n, knotv, knotlen, order );

    matrix P = matrix::Zero( t.size(), knots.size() - 1 );

    for( int i = 0; i < int(knots.size()) - 1; ++i ) {
        const auto low = knots[ i ];
        const auto high = knots[ i + 1 ];
        P.col(i) = (low <= t && t < high).template cast< Numeric >();
    }

    array P1, P2;
    for( int j = 1; j <= order; ++j ) {
        for( int i = 0; i < int(knots.size()) - j; ++i ) {

            const auto low  = knots[i];
            const auto mid  = knots[i + j];
            const auto high = knots[i + j + 1];

            const auto& t1 = (t - low) / (mid - low);
            const auto& t2 = (high - t) / (high - knots[i+1]);

            P1 = P.col( i );
            P2 = P.col( i+1 );

            P.col( i ) = (t1 * P1) + (t2 * P2);
        }
    }

    const auto col = 0;
    const auto colsz = P.cols() - order;
    const auto row = 10 * order;
    const auto rowsz = n;

    return P.block( row, col, rowsz, colsz );
}

struct buffer_guard {
    buffer_guard( Py_buffer& b ) : p( &b ) {}
    ~buffer_guard() { if( this->p ) PyBuffer_Release( this->p ); }

    Py_buffer* p = nullptr;
};

PyObject* bspline( PyObject*, PyObject* args ) {
    Py_buffer bufknots;
    Py_buffer outbuf;
    PyObject* output;
    int samples;
    int order;

    if( !PyArg_ParseTuple( args, "s*Oii",
                                    &bufknots,
                                    &output,
                                    &samples,
                                    &order ) )
        return nullptr;

    buffer_guard g1( bufknots );

    if( PyObject_GetBuffer( output, &outbuf, PyBUF_WRITABLE
                                           | PyBUF_F_CONTIGUOUS
                                           | PyBUF_FORMAT
                ) ) return nullptr;

    buffer_guard g2( outbuf );

    int knotlen = bufknots.len / sizeof( float );
    auto knots = (float*)bufknots.buf;
    auto out = (float*)outbuf.buf;

    auto m = bspline_matrix< float >( samples, knots, knotlen, order );
    Eigen::Map< decltype( m ) >( out, m.rows(), m.cols() ) = m;

    Py_INCREF( output );
    return output;
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
