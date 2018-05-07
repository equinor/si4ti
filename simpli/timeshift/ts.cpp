#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <Python.h>
#include <bytesobject.h>

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>

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

template< typename T >
using vector = Eigen::Matrix< T, Eigen::Dynamic, 1 >;

template< typename T >
using matrix = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

template< typename T >
vector< T > derive( vector< T > signal,
                    vector< T > omega
                  ) {

    /*
     * D = FFTI(iωFFT(signal))
     * where D is the derivative of a signal (data trace)
     *
     * It takes the entire omeg an argument, in order to not recompute it on
     * every invocation (it's shared across all derivations in this
     * application).
     */

    vector< std::complex< T > > ff;
    vector< T > result;


    // TODO: MUST be initialised outside
    static Eigen::FFT< T > fft;
    fft.fwd( ff, signal );
    ff.array() *= std::complex< T >(0, 1) * omega.array();
    fft.inv( result, ff );
    return result;
}

PyObject* pyderive( PyObject*, PyObject* args ) {
    PyObject* signal;
    PyObject* omega;
    PyObject* output;
    Py_buffer sigbuf, omgbuf, outbuf;

    if( !PyArg_ParseTuple( args, "OOO", &signal,
                                        &omega,
                                        &output ) )
        return nullptr;

    if( PyObject_GetBuffer( signal, &sigbuf, PyBUF_ANY_CONTIGUOUS
                                           | PyBUF_ND
                                           | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g1( sigbuf );

    if( PyObject_GetBuffer( omega, &omgbuf, PyBUF_ANY_CONTIGUOUS
                                          | PyBUF_ND
                                          | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g2( omgbuf );

    if( PyObject_GetBuffer( output, &outbuf, PyBUF_WRITABLE
                                           | PyBUF_ANY_CONTIGUOUS
                                           | PyBUF_ND
                                           | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g3( outbuf );

    const auto cols = sigbuf.ndim == 1 ? 1 : sigbuf.shape[1];

    Eigen::Map< matrix< float > > sig( (float*)sigbuf.buf, sigbuf.shape[0], cols );
    Eigen::Map< matrix< float > > omg( (float*)omgbuf.buf, sigbuf.shape[0], cols );
    Eigen::Map< matrix< float > > out( (float*)outbuf.buf, outbuf.shape[0], cols );

    for( int i = 0; i < out.cols(); ++i ) {
        vector< float > s = sig.col(i);
        vector< float > o = omg.col(i);
        out.col(i) = derive( s, o );
    }

    Py_INCREF( output );
    return output;
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
    f.array() /= dt * n;

    return f;
}

template< typename T >
vector< T > angular_frequency( int n, T dt = 1 ) {
    constexpr static const float pi = 3.14159265358979323846f;
    auto f = frequency_spectrum( n, dt );
    f.array() *= 2 * pi;
    return f;
}

PyObject* fftfreq( PyObject*, PyObject* args ) {
    int n;
    float dt;
    PyObject* output;
    Py_buffer out;

    if( !PyArg_ParseTuple( args, "ifO", &n, &dt, &output ) ) return nullptr;

    if( PyObject_GetBuffer( output, &out, PyBUF_WRITABLE
                                        | PyBUF_ANY_CONTIGUOUS
                                        | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g( out );

    auto f = frequency_spectrum( n, dt );
    Eigen::Map< decltype( f ) >( (float*)out.buf, f.rows(), f.cols() ) = f;

    Py_INCREF( output );
    return output;
}

PyObject* angfreq( PyObject*, PyObject* args ) {
    int n;
    float dt;
    PyObject* output;
    Py_buffer out;

    if( !PyArg_ParseTuple( args, "ifO", &n, &dt, &output ) ) return nullptr;

    if( PyObject_GetBuffer( output, &out, PyBUF_WRITABLE
                                        | PyBUF_ANY_CONTIGUOUS
                                        | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g( out );

    auto f = angular_frequency( n, dt );
    Eigen::Map< decltype( f ) >( (float*)out.buf, f.rows(), f.cols() ) = f;

    Py_INCREF( output );
    return output;
}

template< typename T >
std::vector< T > knotvector( int samples, T density ) {
    const auto step = T(1) / density;
    const auto middle = T(samples + 1) / 2;

    std::vector< T > knotv;

    for( auto f = middle; f > 1 / density; f -= step )
        knotv.push_back( f );

    std::reverse( knotv.begin(), knotv.end() );

    for( auto f = middle + step; f < (1+samples) - 1/density; f += step )
        knotv.push_back( f );

    return knotv;
}

PyObject* knotvec( PyObject*, PyObject* args ) {
    int samples;
    float density;
    PyObject* output;
    Py_buffer out;

    if( !PyArg_ParseTuple( args, "ifO", &samples, &density, &output ) )
        return nullptr;

    if( PyObject_GetBuffer( output, &out, PyBUF_WRITABLE
                                        | PyBUF_ANY_CONTIGUOUS
                                        | PyBUF_FORMAT ) )
        return nullptr;

    buffer_guard g( out );

    const auto knotv = knotvector( samples, density );
    std::copy( knotv.begin(), knotv.end(), (float*)out.buf );

    Py_INCREF( output );
    return output;
}

PyObject* spline( PyObject*, PyObject* args ) {
    int samples;
    float density;
    int degree;
    PyObject* output;
    Py_buffer out;

    if( !PyArg_ParseTuple( args, "ifiO", &samples,
                                         &density,
                                         &degree,
                                         &output ) )
        return nullptr;

    if( PyObject_GetBuffer( output, &out, PyBUF_WRITABLE
                                        | PyBUF_ANY_CONTIGUOUS
                                        | PyBUF_FORMAT ) )
        return nullptr;

    buffer_guard g( out );

    const auto knotv = knotvector( samples, density );
    auto m = bspline_matrix( samples, knotv.data(), knotv.size(), degree );
    vector< float > colsums = m.colwise().sum();
    colsums.array() = 1.0 / colsums.array();
    m *= colsums.asDiagonal();
    Eigen::Map< decltype( m ) >( (float*)out.buf, m.rows(), m.cols() ) = m;

    Py_INCREF( output );
    return output;
}

template< typename T >
matrix< T > constraints( const matrix< T >& spline,
                         T vertical_smoothing,
                         T horizontal_smoothing ) {

    /*
     * Smoothing constraints
     *
     * MxM matrix (M = number of spline functions) containing the vertical
     * smoothing and central component of the horizontal smoothing.
     *
     */

    auto basis_squared = spline.transpose() * spline;
    const auto tracelen = spline.rows();

    /*
     * The matrix D computes an array of differences d[n] - d[n+1] representing
     * level of vertical smoothness.
     *
     * It is a tracelen-1 x tracelen matrix on the form:
     *
     *               1 -1  0 .. 0  0
     *               0  1 -1 .. 0  0
     *        D  =   .  .  . .. .  .
     *               .  .  . .. .  .
     *               0  0  0 .. 1 -1
     */
    matrix< T > D = matrix< T >::Identity( tracelen - 1, tracelen );
    D.template diagonal< 1 >().fill( -1.0 );
    matrix< T > Dpm = spline.transpose() * D.transpose() * D * spline;

    return (horizontal_smoothing * basis_squared) + (vertical_smoothing * Dpm);
}

PyObject* pyconstr( PyObject*, PyObject* args ) {
    PyObject* spline;
    PyObject* output;
    Py_buffer splinebuf, outbuf;
    float vertical_smoothing;
    float horizontal_smoothing;

    if( !PyArg_ParseTuple( args, "OffO", &spline,
                                         &vertical_smoothing,
                                         &horizontal_smoothing,
                                         &output ) )
        return nullptr;

    if( PyObject_GetBuffer( spline, &splinebuf, PyBUF_ANY_CONTIGUOUS
                                              | PyBUF_ND
                                              | PyBUF_FORMAT
                          ) )
        return nullptr;

    buffer_guard g1( splinebuf );

    if( PyObject_GetBuffer( output, &outbuf, PyBUF_WRITABLE
                                           | PyBUF_ANY_CONTIGUOUS
                                           | PyBUF_ND
                                           | PyBUF_FORMAT
                ) ) return nullptr;

    buffer_guard g2( outbuf );

    Eigen::Map< matrix< float > > sp( (float*)splinebuf.buf,
                                        splinebuf.shape[1],
                                        splinebuf.shape[0]
                                      );

    matrix< float > spli = sp;

    Eigen::Map< matrix< float > > out( (float*)outbuf.buf,
                                        outbuf.shape[0],
                                        outbuf.shape[1]
                                     );

    out = constraints( spli, vertical_smoothing, horizontal_smoothing );

    Py_INCREF( output );
    return output;
}

template< typename T >
matrix< T > linearoperator( const vector< T >& derived,
                            const matrix< T >& spline ) {
    /*
     * Compute the linear operator
     *
     * Returns an MxM matrix, where M is number of spline functions
     */
    auto Lt = derived.asDiagonal() * spline;
    return Lt.transpose() * Lt;
}

PyObject* linop( PyObject*, PyObject* args ) {
    PyObject* derived;
    PyObject* spline;
    PyObject* output;
    Py_buffer derivbuf, splinebuf, outbuf;

    if( !PyArg_ParseTuple( args, "OOO", &derived,
                                        &spline,
                                        &output
                                        ) )
        return nullptr;

    if( PyObject_GetBuffer( derived, &derivbuf, PyBUF_ANY_CONTIGUOUS
                                              | PyBUF_ND
                                              | PyBUF_FORMAT
                                        ) )
        return nullptr;

    buffer_guard g1( derivbuf );

    if( PyObject_GetBuffer( spline, &splinebuf, PyBUF_ANY_CONTIGUOUS
                                              | PyBUF_ND
                                              | PyBUF_FORMAT
                                        ) )
        return nullptr;

    buffer_guard g2( splinebuf );

    if( PyObject_GetBuffer( output, &outbuf, PyBUF_WRITABLE
                                           | PyBUF_ANY_CONTIGUOUS
                                           | PyBUF_ND
                                           | PyBUF_FORMAT
                ) )
        return nullptr;

    buffer_guard g3( outbuf );

    Eigen::Map< vector< float > > dv( (float*)derivbuf.buf,
                                        derivbuf.shape[0]
                                    );


    Eigen::Map< matrix< float > > sp( (float*)splinebuf.buf,
                                        splinebuf.shape[1],
                                        splinebuf.shape[0]
                                    );

    vector< float > deriv = dv;
    matrix< float > spli = sp;

    Eigen::Map< matrix< float > > out( (float*)outbuf.buf,
                                        outbuf.shape[0],
                                        outbuf.shape[1]
                                     );

    out = linearoperator( deriv, spli );

    Py_INCREF( output );
    return output;
}


PyMethodDef methods[] = {
    { "bspline", (PyCFunction) bspline,  METH_VARARGS, "B-spline as matrix." },
    { "derive",  (PyCFunction) pyderive, METH_VARARGS, "Derive with FFT." },
    { "fftfreq", (PyCFunction) fftfreq,  METH_VARARGS, "Frequency spectrum." },
    { "angfreq", (PyCFunction) angfreq,  METH_VARARGS, "Angular frequency." },
    { "knotvec", (PyCFunction) knotvec,  METH_VARARGS, "Knot vector." },
    { "spline",  (PyCFunction) spline,   METH_VARARGS, "Spline." },
    { "constr",  (PyCFunction) pyconstr, METH_VARARGS, "Constraints matrix." },
    { "linop",   (PyCFunction) linop,    METH_VARARGS, "Linear operator." },
    { nullptr }
};

}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_ts( void ) {

    static struct PyModuleDef bspline_module = {
            PyModuleDef_HEAD_INIT,
            "ts",   /* name of module */
            NULL,
            -1,
            methods,
    };
    return PyModule_Create( &bspline_module );
}

#else

PyMODINIT_FUNC initts( void ) {
    (void)Py_InitModule( "ts", methods );
}

#endif // PY3
