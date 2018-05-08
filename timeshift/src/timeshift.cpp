#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <getopt.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <segyio/segy.h>

namespace {

struct options {
    std::vector< std::string > files;
    double timeshift_resolution = 0.05;
    double horizontal_smoothing = 0.01;
    double vertical_smoothing   = 0.1;
    int    solver_max_iter      = 100;
    bool   double_precision     = false;
    bool   correct_4d_noise     = false;
    double normalizer           = 30;
    int    verbosity            = 0;
    int    ilbyte               = SEGY_TR_INLINE;
    int    xlbyte               = SEGY_TR_CROSSLINE;
};

options parseopts( int argc, char** argv ) {
    static struct option longopts[] = {
        { "timeshift-resolution", required_argument, 0, 'r' },
        { "horizontal-smoothing", required_argument, 0, 'H' },
        { "vertical-smoothing",   required_argument, 0, 'V' },
        { "max-iter",             required_argument, 0, 'm' },
        { "double-precision",     no_argument,       0, 'd' },
        { "correct-4D",           no_argument,       0, 'c' },
        { "normalizer",           required_argument, 0, 'n' },
        { "ilbyte",               required_argument, 0, 'i' },
        { "xlbyte",               required_argument, 0, 'x' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    options opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "r:H:V:m:dcn:i:x:v",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'r': break;
            case 'H': break;
            case 'V': break;
            case 'm': break;
            case 'd': break;
            case 'c': break;
            case 'n': break;
            case 'i': opts.ilbyte = std::stoi( optarg ); break;
            case 'x': opts.xlbyte = std::stoi( optarg ); break;
            case 'v': break;
            case 'h': break;

            default:
                std::exit( 1 );
        }
    }

    if( argc - optind < 2 ) {
        std::cerr << "Need at least 2 input files\n";
        std::exit( 1 );
    }

    while( optind < argc )
        opts.files.push_back( argv[ optind++ ] );

    return opts;
}

using filehandle = std::unique_ptr< segy_file, decltype( &segy_close ) >;

filehandle openfile( const std::string& fname, std::string mode = "rb" ) {
    if( mode.back() != 'b' ) mode.push_back( 'b' );

    filehandle fp( segy_open( fname.c_str(), mode.c_str() ), &segy_close );
    if( !fp ) throw std::invalid_argument( "Unable to open file " + fname );

    return fp;
}

struct geometry {
    int samples;
    int traces;
    int trace_bsize;
    long trace0;
};

geometry findgeometry( segy_file* fp ) {
    // TODO: proper error handling other than throw exceptions without checking
    // what went wrong
    char header[ SEGY_BINARY_HEADER_SIZE ] = {};

    auto err = segy_binheader( fp, header );
    if( err != SEGY_OK ) throw std::runtime_error( "Unable to read header" );

    geometry geo;
    geo.samples     = segy_samples( header );
    geo.trace0      = segy_trace0( header );
    geo.trace_bsize = segy_trace_bsize( geo.samples );

    err = segy_traces( fp, &geo.traces, geo.trace0, geo.trace_bsize );

    if( err != SEGY_OK )
        throw std::runtime_error( "Could not compute trace count" );

    return geo;
}

template< typename Vector >
void writefile( segy_file* basefile,
                const Vector& v,
                const std::string& fname,
                const geometry& geo ) {

    char textheader[ SEGY_TEXT_HEADER_SIZE ];
    char binaryheader[ SEGY_BINARY_HEADER_SIZE ];
    char traceheader[ SEGY_TRACE_HEADER_SIZE ];
    int err = 0;

    auto fp = openfile( fname.c_str(), "w+b" );

    err = segy_read_textheader( basefile, textheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to read text header" );
    err = segy_write_textheader( fp.get(), 0, textheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to write text header" );

    err = segy_binheader( basefile, binaryheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to read binary header" );
    err = segy_write_binheader( fp.get(), binaryheader );
    if( err != SEGY_OK )
        throw std::runtime_error( "Unable to write binary header" );

    auto buf = v.data();
    for( int traceno = 0; traceno < geo.traces; ++traceno ) {
        err = segy_traceheader( basefile,
                                traceno,
                                traceheader,
                                geo.trace0,
                                geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to read trace header "
                                    + std::string( traceno ) );

        err = segy_write_traceheader( fp.get(),
                                      traceno,
                                      traceheader,
                                      geo.trace0,
                                      geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to write trace header "
                                    + std::string( traceno ) );

        err = segy_writetrace( fp.get(),
                               traceno,
                               buf,
                               geo.trace0,
                               geo.trace_bsize );

        if( err != SEGY_OK )
            throw std::runtime_error( "Unable to write trace "
                                    + std::string( traceno ) );

        buf += geo.samples;
    }
}

}

int main( int argc, char** argv ) {
    auto opts = parseopts( argc, argv );

    std::vector< filehandle > filehandles;
    std::vector< geometry > geometries;
    for( const auto& file : opts.files ) {
        filehandles.push_back( openfile( file ) );
        geometries.push_back( findgeometry( filehandles.back().get() ) );
    }
}
