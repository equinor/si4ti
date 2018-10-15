#include "impedance.hpp"

#include <getopt.h>

int Progress::count = 0;
int Progress::expected = 10;

namespace {

void printhelp(){
    std::cout <<
        "Usage: timeshift [OPTION]... [FILE]...\n"
        "Compute relative acoustic impedance from seismic vintages.\n"
        "\n"
        "-t, --timevarying-wavelet     use a windowed time varying wavelet\n"
        "-l, --lateral-smoothing-3D    horizontal smoothing factor\n"
        "-L, --lateral-smoothing-4D    4D extension of the horizontal\n"
        "                              smoothing. This will give preference to\n"
        "                              a solution with similar lateral\n"
        "                              smoothness at corresponding points on\n"
        "                              the vintages\n"
        "-d, --damping-3D              constrains the relative acoustic\n"
        "                              impedance on each vintage so it does\n"
        "                              not deviate too much from zero\n"
        "-D, --damping-4D              constraint on the difference in\n"
        "                              relative acoustic impedance between\n"
        "                              vintages\n"
        "-s, --segments                if this parameter is set, data domain\n"
        "                              splitting will be performed. Takes the\n"
        "                              number of segments as an argument.\n"
        "-p, --inverse-polarity        invert polarity of the data\n"
        "-m, --max-iter                maximum number of itarations for\n"
        "                              linear solver\n"
        "-i, --ilbyte                  inline header word byte offset\n"
        "-x, --xlbyte                  crossline header word byte offset\n"
        "-v, --verbose                 increase verbosity\n"
        "-h, --help                    display this help and exit\n"
        "\n\n"
    ;
}

struct options {
    std::vector< std::string > files;
    int             verbosity            = 0;
    segyio::ilbyte  ilbyte               = segyio::ilbyte();
    segyio::xlbyte  xlbyte               = segyio::xlbyte();
    int             polarity             = 1;
    int             segments             = 1;
    bool            tv_wavelet           = false;
    double          damping_3D           = 0.0001;
    double          damping_4D           = 0.0001;
    double          latsmooth_3D         = 0.05;
    double          latsmooth_4D         = 4;
    int             max_iter             = 50;
};

options parseopts( int argc, char** argv ) {
    static struct option longopts[] = {
        { "ilbyte",               required_argument, 0, 'i' },
        { "xlbyte",               required_argument, 0, 'x' },
        { "inverse-polarity",     no_argument,       0, 'p' },
        { "timevarying-wavelet",  no_argument,       0, 't' },
        { "damping-3D",           required_argument, 0, 'd' },
        { "damping-4D",           required_argument, 0, 'D' },
        { "lateral-smoothing-3D", required_argument, 0, 'l' },
        { "lateral-smoothing-4D", required_argument, 0, 'L' },
        { "segments",             required_argument, 0, 's' },
        { "max-iter",             required_argument, 0, 'm' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    options opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "i:x:ptd:D:l:L:s:m:vh",
                             longopts, &option_index );

        if( c == -1 ) break;

        switch( c ) {
            case 'i':
                opts.ilbyte = segyio::ilbyte( std::stoi( optarg ) );
                break;
            case 'x':
                opts.xlbyte = segyio::xlbyte( std::stoi( optarg ) );
                break;
            case 'p': opts.polarity     = -1; break;
            case 't': opts.tv_wavelet   = true; break;
            case 'd': opts.damping_3D   = std::stod( optarg ); break;
            case 'D': opts.damping_4D   = std::stod( optarg ); break;
            case 'l': opts.latsmooth_3D = std::stod( optarg ); break;
            case 'L': opts.latsmooth_4D = std::stod( optarg ); break;
            case 's': opts.segments     = std::stoi( optarg ); break;
            case 'm': opts.max_iter     = std::stoi( optarg ); break;
            case 'v': opts.verbosity++;                        break;
            case 'h':
                printhelp();
                std::exit( 0 );
            default:
                printhelp();
                std::exit( 1 );
        }
    }

    if( argc - optind < 1 ) {
        std::cerr << "Need at least 1 input file\n";
        std::exit( 1 );
    }

    while( optind < argc )
        opts.files.push_back( argv[ optind++ ] );

    return opts;
}

output_file create_file( std::string filename,
                         std::string basefile,
                         segyio::ilbyte ilbyte, segyio::xlbyte xlbyte ) {

    std::ifstream in( basefile );
    std::ofstream dst( filename );
    dst << in.rdbuf();
    dst.close();

    return output_file( segyio::path{ filename },
                        segyio::config{}.with( ilbyte )
                                        .with( xlbyte ) );
}

}

int main( int argc, char** argv ) {
    using T = float;

    auto opts = parseopts( argc, argv );

    Progress::expected += opts.segments * ( opts.max_iter + 25 );

    std::vector< input_file > files;

    for( const auto& file : opts.files )
        files.emplace_back( segyio::path{ file },
                            segyio::config{}.with( opts.ilbyte )
                                            .with( opts.xlbyte ) );

    for( auto& file: files )
        if( not ( file.sorting() == segyio::sorting::iline() ) )
            throw std::runtime_error("File must be inline sorted");

    std::vector< matrix< T > > wvlets = wavelets< T >( files,
                                                       opts.tv_wavelet,
                                                       opts.polarity );

    Progress::report( 5 );

    T norm = normalization( wvlets );
    const int vintages = files.size();

    std::vector< matrix< T > > A = forward_operators< T >( wvlets,
                                                           vintages,
                                                           norm );

    std::vector< output_file > relAI_files;
    std::vector< output_file > dsyn_files;

    for( int i = 0; i < vintages; ++i ) {
        std::string relAI_fname = "relAI-" + std::to_string( i ) + ".sgy";
        std::string dsyn_fname = "dsyn-" + std::to_string( i ) + ".sgy";

        auto relAI_file = create_file( relAI_fname,
                                       opts.files.front(),
                                       opts.ilbyte,
                                       opts.xlbyte );
        auto dsyn_file = create_file( dsyn_fname,
                                      opts.files.front(),
                                      opts.ilbyte,
                                      opts.xlbyte );

        relAI_files.emplace_back( std::move( relAI_file ) );
        dsyn_files.emplace_back( std::move( dsyn_file ) );
    }

    const int ilines = files.front().inlinecount();
    const int xlines = files.front().crosslinecount();
    const int tracelen = files.front().samplecount();

    const auto sgments = segments( opts.segments,
                                   ilines, xlines,
                                   opts.max_iter );

    for( const auto& segment : sgments ) {
        const int trc_start = segment.first;
        const int trc_end = segment.second;

        vector< T > relAI = compute_impedance< T >( files,
                                                    relAI_files,
                                                    A,
                                                    norm,
                                                    opts.max_iter,
                                                    opts.damping_3D,
                                                    opts.damping_4D,
                                                    opts.latsmooth_3D,
                                                    opts.latsmooth_4D,
                                                    trc_start, trc_end );

        const int traces = trc_end - trc_start + 1;
        const int cubesize = traces * tracelen;

        for( int i = 0; i < vintages; ++i ) {
            auto seg = relAI.segment( i * cubesize, cubesize );

            writefile( seg,
                       relAI_files[ i ],
                       trc_start, trc_end );

            seg = reconstruct_data< T >( seg, A[ i ], norm, traces );

            writefile( seg,
                       dsyn_files[ i ],
                       trc_start, trc_end );
        }
        Progress::report( 5 );
    }
}
