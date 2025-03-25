#include "impedance.hpp"

#include <getopt.h>

#include <segyio/segyio.hpp>

using namespace segyio::literals;

template< typename Derived >
struct xlinesorted_checker {
    bool xlinesorted() const noexcept(true);
};

template< typename Derived >
bool xlinesorted_checker< Derived >::xlinesorted() const noexcept(true) {
    const auto* self = static_cast< const Derived* >( this );
    return self->sorting() == segyio::sorting::xline();
}


using input_file = segyio::basic_volume< segyio::readonly,
                                         xlinesorted_checker >;
using output_file = segyio::basic_volume< segyio::trace_writer,
                                          segyio::write_always,
                                          xlinesorted_checker >;

int Progress::count = 0;
int Progress::expected = 10;

namespace {

struct options {
    std::vector< std::string > files;
    std::vector< std::string > output_files;
    int             verbosity            = 0;
    segyio::ilbyte  ilbyte               = segyio::ilbyte();
    segyio::xlbyte  xlbyte               = segyio::xlbyte();
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
        "-o, --overlap                 number of inlines (crosslines if\n"
        "                              crossline sorted) overlap between\n"
        "                              segments when performing data domain\n"
        "                              splitting. Defaults to maximum number\n"
        "                              of iterations of the linear solver \n"
        "-p, --inverse-polarity        invert polarity of the data\n"
        "-m, --max-iter                maximum number of iterations for\n"
        "                              linear solver\n"
        "-O, --output-files            space separated list of filenames. The\n"
        "                              list is terminated by a double dash\n"
        "                              [out1.sgy ... --]. Relative acoustic\n"
        "                              impedance will be output to the first\n"
        "                              files, and the synthetic data to the\n"
        "                              following files\n"
        "-i, --ilbyte                  inline header word byte offset\n"
        "-x, --xlbyte                  crossline header word byte offset\n"
        "-v, --verbose                 increase verbosity\n"
        "-h, --help                    display this help and exit\n"
        "\n\n"
    ;
}

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
        { "overlap",              required_argument, 0, 'o' },
        { "max-iter",             required_argument, 0, 'm' },
        { "output-files",         required_argument, 0, 'O' },
        { "verbose",              no_argument,       0, 'v' },
        { "help",                 no_argument,       0, 'h' },
        { nullptr },
    };

    options opts;

    while( true ) {
        int option_index = 0;
        int c = getopt_long( argc, argv,
                             "i:x:ptd:D:l:L:s:o:m:O:vh",
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
            case 'o': opts.overlap      = std::stoi( optarg ); break;
            case 'm': opts.max_iter     = std::stoi( optarg ); break;
            case 'v': opts.verbosity++;                        break;
            case 'O':
                optind--;
                while( "--" != std::string( argv[optind] ) )
                    opts.output_files.push_back( argv[optind++] );
                optind++;
                break;
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

    if( 2*opts.files.size() != opts.output_files.size()
        and !opts.output_files.empty() ) {
        std::cerr << "The number of output files should be twice the number "
                  << "of input files.";
        std::exit( 1 );
    }

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
    auto opts = parseopts( argc, argv );

    std::vector< input_file > files;

    for( const auto& file : opts.files ) {
        files.emplace_back( segyio::path{ file },
                            segyio::config{}.with( opts.ilbyte )
                                            .with( opts.xlbyte ) );

        auto& back = files.back();
        auto& front = files.front();

        if( not (front.sorting()        == back.sorting())
            or   front.crosslinecount() != back.crosslinecount()
            or   front.inlinecount()    != back.inlinecount()
            or   front.tracecount()     != back.tracecount() )

            throw std::invalid_argument( "Input files must all "
                                         "have equal structure" );
    }

    std::vector< output_file > relAI_files;
    std::vector< output_file > dsyn_files;

    const int vintages = files.size();
    for( int i = 0; i < vintages; ++i ) {

        std::string relAI_fname;
        std::string dsyn_fname;

        if( opts.output_files.empty() ) {
            relAI_fname = "relAI-" + std::to_string( i ) + ".sgy";
            dsyn_fname = "dsyn-" + std::to_string( i ) + ".sgy";
        }
        else {
            relAI_fname = opts.output_files[ i ];
            dsyn_fname = opts.output_files[ i + vintages ];
        }

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

    compute_impedance_of_full_cube(files, relAI_files, dsyn_files, opts);

}
