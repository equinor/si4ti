#include <string>
#include <vector>

#include <getopt.h>

#include <Eigen/Core>

#include "timeshift.hpp"

#define EIGEN_DONT_PARALLELIZE

int Progress::count = 0;
int Progress::expected = 60;

namespace {

void printhelp(){
    std::cout <<
        "Usage: timeshift [OPTION]... [FILE]...\n"
        "Compute timeshifts between seismic vintages.\n"
        "\n"
        "-r, --timeshift-resolution    downscale the resolution of the\n"
        "                              timeshift by this factor\n"
        "-H, --horizontal-smoothing    horizontal smoothing factor\n"
        "-V, --vertical-smoothing      vertical smoothing factor\n"
        "-m, --max-iter                maximum number of itarations for \n"
        "                              linear solver\n"
        "-d, --double-precision        use double precision numeric type for\n"
        "                              computations\n"
        "-c, --correct-4D              apply 4D correction\n"
        "-s, --cumulative              output comulative timeshifts.\n"
        "-S, --scaling                 Data is normalized and scaled by this\n"
        "                              factor. Defaults to 30\n"
        "-A, --normalization           Normalization factor. Normalize the\n"
        "                              data by this factor. By default this\n"
        "                              is computed from the input cubes\n"
        "-N, --output-normalization    output normalization during run\n"
        "-C, --compute-normalization   Compute normalization (and multiply by\n"
        "                              scaling), write to stdout and exit\n"
        "-P, --output-dir              output directory\n"
        "-p, --output-prefix           output filename prefix\n"
        "-D, --output-delim            output filename delimiter\n"
        "-X, --output-ext              output file extension\n"
        "-o, --output-files            space separated list of filenames. The\n"
        "                              list is terminated by a double dash\n"
        "                              [out1.sgy ... --]. If this option is\n"
        "                              set, all other filename specifiers\n"
        "                              will be ignored (-P, -p, -D,-X).\n"
        "-i, --ilbyte                  inline header word byte offset\n"
        "-x, --xlbyte                  crossline header word byte offset\n"
        "-t, --sampling-interval       sampling interval\n"
        "-v, --verbose                 increase verbosity\n"
        "-h, --help                    display this help and exit\n"
        "\n\n"
    ;
}

segyio::ilbyte mkilbyte( const std::string& optarg ) {
    return segyio::ilbyte{ std::stoi( optarg ) };
}

segyio::xlbyte mkxlbyte( const std::string& optarg ) {
    return segyio::xlbyte{ std::stoi( optarg ) };
}

options parseopts( int argc, char** argv ) {
    static struct option longopts[] = {
        { "timeshift-resolution", required_argument, 0, 'r' },
        { "horizontal-smoothing", required_argument, 0, 'H' },
        { "vertical-smoothing",   required_argument, 0, 'V' },
        { "max-iter",             required_argument, 0, 'm' },
        { "double-precision",     no_argument,       0, 'd' },
        { "correct-4D",           no_argument,       0, 'c' },
        { "cumulative",           no_argument,       0, 's' },
        { "scaling",              required_argument, 0, 'S' },
        { "normalization",        required_argument, 0, 'A' },
        { "output-normalization", no_argument,       0, 'N' },
        { "compute-normalization",no_argument,       0, 'C' },
        { "output-dir",           required_argument, 0, 'P' },
        { "output-prefix",        required_argument, 0, 'p' },
        { "output-delim",         required_argument, 0, 'D' },
        { "output-ext",           required_argument, 0, 'X' },
        { "output-files",         required_argument, 0, 'o' },
        { "ilbyte",               required_argument, 0, 'i' },
        { "xlbyte",               required_argument, 0, 'x' },
        { "sampling-interval",    required_argument, 0, 't' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    options opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "r:H:V:m:dcsNCS:A:P:p:D:X:o:i:x:t:v",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'r': opts.timeshift_resolution = std::stod( optarg ); break;
            case 'H': opts.horizontal_smoothing = std::stod( optarg ); break;
            case 'V': opts.vertical_smoothing   = std::stod( optarg ); break;
            case 'm': opts.solver_max_iter      = std::stoi( optarg ); break;
            case 'd': opts.double_precision     = true; break;
            case 'c': opts.correct_4d_noise     = true; break;
            case 's': opts.cumulative           = true; break;
            case 'S': opts.scaling              = std::stod( optarg ); break;
            case 'A': opts.normalization        = std::stod( optarg ); break;
            case 'N': opts.output_norm          = true; break;
            case 'C': opts.compute_norm         = true; break;
            case 'P': opts.dir                  = optarg; break;
            case 'p': opts.prefix               = optarg; break;
            case 'D': opts.delim                = optarg; break;
            case 'X': opts.extension            = optarg; break;
            case 'o':
                optind--;
                while( "--" != std::string( argv[optind] ) )
                    opts.output_files.push_back( argv[optind++] );
                optind++;
                break;
            case 't': opts.sampling_interval    = std::stod( optarg ); break;
            case 'i': opts.ilbyte               = mkilbyte( optarg ); break;
            case 'x': opts.xlbyte               = mkxlbyte( optarg ); break;
            case 'v': break;
            case 'h':
                printhelp();
                std::exit( 0 );

            default:
                printhelp();
                std::exit( 1 );
        }
    }

    if( argc - optind < 2 ) {
        std::cerr << "Need at least 2 input files\n";
        std::exit( 1 );
    }

    while( optind < argc )
        opts.files.push_back( argv[ optind++ ] );

    if( !opts.extension.empty() and opts.extension.front() != '.' )
        opts.extension.insert( opts.extension.begin(), '.' );

    if( !opts.dir.empty() and opts.dir.back() != '/' )
        opts.dir.push_back( '/' );

    return opts;
}

template< typename T >
void run( const options& opts ) try {
    std::vector< input_file > files;
    std::vector< geometry > geometries;
    for( const auto& file : opts.files ) {
        files.push_back( { segyio::path{ file },
                           segyio::config{}.with( opts.ilbyte )
                                           .with( opts.xlbyte ) } );

        geometries.push_back( findgeometry( files.back() ) );
    }

    const double sample_interval = infer_interval( files[0],
                                                   opts.sampling_interval );
    if( opts.compute_norm ){
        std::cout << normalize_surveys( opts.scaling, files ) << "\n";
        std::exit( 0 );
    }

    const auto vintages = files.size();
    const auto samples = geometries.back().samples;
    const int splineord = 3;
    const auto B = normalized_bspline( samples,
                                       T( opts.timeshift_resolution ),
                                       splineord );

    auto x = compute_timeshift( B, splineord, files, geometries, opts );

    const auto M = B.cols() * geometries.front().traces;
    const int timeshifts = vintages - 1;

    std::vector< std::string > fnames;

    if( opts.output_files.empty() ) {
        for( int i = 0; i < timeshifts; ++i ) {
            const auto fname = opts.dir
                             + opts.prefix
                             + opts.delim
                             + std::to_string( i )
                             + opts.extension
                             ;
            fnames.push_back( fname );
        }
    }
    else {
        if( opts.output_files.size() != timeshifts ) {
            std::cerr << "The number of output files should be one less than "
                      << "the number of input files\n";
            std::exit( 1 );
        }

        fnames = opts.output_files;
    }

    for( int i = 0; i < timeshifts; ++i ) {
        vector< T > seg = x.segment( i * M, M );

        output_timeshift< T >( opts.files.front(),
                               seg,
                               fnames[ i ],
                               geometries.back(),
                               sample_interval,
                               B );
    }
    Progress::report( 5 );
} catch( std::exception& e ) {
    std::cerr << e.what() << '\n';
}

}

int main( int argc, char** argv ) {
    Eigen::initParallel();
    auto opts = parseopts( argc, argv );

    Progress::expected += opts.solver_max_iter;

    if( opts.correct_4d_noise ) Progress::expected += opts.solver_max_iter + 20;

    if( not opts.double_precision ) run< float >( opts );
    else run< double >( opts );
}
