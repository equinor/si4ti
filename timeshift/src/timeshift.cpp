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
        "                              scaling), write to stoud and exit\n"
        "-P, --output-dir              output directory\n"
        "-p, --output-prefix           output filename prefix\n"
        "-D, --output-delim            output filename delimiter\n"
        "-X, --output-ext              output file extension\n"
        "-i, --ilbyte                  inline header word byte offset\n"
        "-x, --xlbyte                  crossline header word byte offset\n"
        "-v, --verbose                 increase verbosity\n"
        "-h, --help                    display this help and exit\n"
        "\n\n"
    ;
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
        { "output-normalization", required_argument, 0, 'N' },
        { "compute-normalization",required_argument, 0, 'C' },
        { "output-dir",           required_argument, 0, 'P' },
        { "output-prefix",        required_argument, 0, 'p' },
        { "output-delim",         required_argument, 0, 'D' },
        { "output-ext",           required_argument, 0, 'X' },
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
                             "r:H:V:m:dcsNCS:A:P:p:D:i:x:v",
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
            case 'i': opts.ilbyte = std::stoi( optarg ); break;
            case 'x': opts.xlbyte = std::stoi( optarg ); break;
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
void run( const options& opts ) {
    std::vector< sio::simple_file > files;
    std::vector< geometry > geometries;
    for( const auto& file : opts.files ) {
        files.push_back( { file, sio::config().ilbyte( opts.ilbyte )
                                              .xlbyte( opts.xlbyte ) }  );

        geometries.push_back( findgeometry( files.back() ) );
    }

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

    auto reconstruct = [&]( vector< T > seg ) {
        const auto scale = 4.0;
        const auto samples = geometries.front().samples;
        const auto M = B.cols();

        vector< T > reconstructed( geometries.front().traces * samples );
        for( int i = 0; i < geometries.front().traces; ++i ) {
            reconstructed.segment( i * samples, samples )
                = scale * B * seg.segment( i * M, M );
        }

        return reconstructed;
    };

    const auto M = B.cols() * geometries.front().traces;
    const int timeshifts = vintages - 1;

    std::vector< std::string > fnames;
    for( int i = 0; i < timeshifts; ++i ) {
        const auto fname = opts.dir
                         + opts.prefix
                         + opts.delim
                         + std::to_string( i )
                         + opts.extension
                         ;
        fnames.push_back( fname );
    }

    for( int i = 0; i < timeshifts; ++i ) {
        vector< T > seg = x.segment( i * M, M );
        auto timeshift = reconstruct( seg );

        writefile( opts.files.front(),
                   timeshift,
                   fnames[ i ],
                   geometries.back() );
    }
    Progress::report( 5 );
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
