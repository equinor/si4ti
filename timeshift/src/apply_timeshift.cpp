#include <iostream>
#include <cstdlib>
#include <memory> // Necessary for segyio, should it be?
#include <algorithm>

#include <getopt.h>

#include <Eigen/Core>
#include <segyio/segyio.hpp>

#include "timeshift.hpp"

namespace {

void printhelp(){
    std::cout <<
        "Usage: timeshift [OPTION] CUBE TIMESHIFT\n"
        "Apply timeshift to a cube.\n"
        "\n"
        "-O, --output-dir              output directory\n"
        "-o, --output                  output filename\n"
        "-i, --ilbyte                  inline header word byte offset\n"
        "-x, --xlbyte                  crossline header word byte offset\n"
        "-r, --reverse                 reverse the timeshift before applying\n"
        "-v, --verbose                 increase verbosity\n"
        "-h, --help                    display this help and exit\n"
        "\n\n"
    ;
}

struct opt {
    std::vector< std::string > files;
    std::string output_dir           = "./";
    std::string output_filename      = "shifted.sgy";
    int         verbosity            = 0;
    int         ilbyte               = SEGY_TR_INLINE;
    int         xlbyte               = SEGY_TR_CROSSLINE;
    int         reverse              = -1;
    double      sampling_interval    = 4.0;
};

opt parseopts( int argc, char** argv ) {
    static struct option longopts[] = {
        { "output-dir",           required_argument, 0, 'O' },
        { "output",               required_argument, 0, 'o' },
        { "ilbyte",               required_argument, 0, 'i' },
        { "xlbyte",               required_argument, 0, 'x' },
        { "reverse",              no_argument,       0, 'r' },
        { "sampling-interval",    required_argument, 0, 's' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    opt opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "O:o:i:x:rs:vh",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'O': opts.output_dir        = optarg; break;
            case 'o': opts.output_filename   = optarg; break;
            case 'i': opts.ilbyte            = std::stoi( optarg ); break;
            case 'x': opts.xlbyte            = std::stoi( optarg ); break;
            case 'r': opts.reverse           = 1; break;
            case 's': opts.sampling_interval = std::stod( optarg ); break;
            case 'v': opts.verbosity++; break;
            case 'h':
                printhelp();
                std::exit( 0 );
        }
    }

    if( argc - optind != 2 ) {
        printhelp();
        std::exit( 1 );
    }

    while( optind < argc )
        opts.files.push_back( argv[ optind++ ] );

    return opts;
}

}

int main( int argc, char** argv ) {
    auto opts = parseopts( argc, argv );

    const auto cube_fname =  opts.files[0];
    const auto shift_fname =  opts.files[1];
    const auto out_fname =  opts.output_dir + opts.output_filename;

    std::ifstream in( cube_fname );
    std::ofstream dst( out_fname );
    dst << in.rdbuf();
    dst.close();

    sio::simple_file cube( cube_fname,
                           sio::config().ilbyte( opts.ilbyte )
                                        .xlbyte( opts.xlbyte )
                         );
    sio::simple_file timeshift( shift_fname,
                                sio::config().ilbyte( opts.ilbyte )
                                             .xlbyte( opts.xlbyte )
                              );
    sio::simple_file shifted_cube( out_fname,
                                   sio::config().ilbyte( opts.ilbyte )
                                                .xlbyte( opts.xlbyte )
                                                .readwrite()
                                 );

    const int samples = cube.read(0).size();
    const int traces = cube.inlines().size() * cube.crosslines().size();

    Eigen::VectorXd trace( samples );
    Eigen::VectorXd shift( samples );
    for( int trc = 0; trc < traces; ++trc ) {
        cube.read( trc, trace.data() );
        timeshift.read( trc, shift.data() );
        shift *= opts.reverse / opts.sampling_interval;
        apply_timeshift( trace, shift );
        shifted_cube.put( trc, trace.data() );
    }

}
