// ======================================================================
// \title  Main.cpp
// \brief main program for the F' application. Intended for CLI-based systems (Linux, macOS)
//
// ======================================================================
// Used to access topology functions
#include <flight/PolarisFsw/Top/PolarisFswTopology.hpp>
// OSAL initialization
#include <Os/Os.hpp>
// Used for signal handling shutdown
#include <signal.h>
// Used for command line argument processing
#include <getopt.h>
// Used for atoi
#include <cstdlib>
// Used for logging to the console
#include <Fw/Logger/Logger.hpp>

/**
 * \brief print command line help message
 *
 * This will print a command line help message including the available command line arguments.
 *
 * @param app: name of application
 */
void print_usage(const char* app) {
  Fw::Logger::log(
      "Usage: ./%s [options]\n"
      "-a\thostname/IP address (GDS ground link)\n"
      "-p\tport_number (GDS ground link)\n"
      "-s\tSITL lockstep port (connects to the truth sim on 127.0.0.1; "
      "0/absent = SITL disabled)\n"
      "-c\tenable the ScriptedCmdSource actuator profile (SITL only; "
      "default off = zero commands)\n"
      "-E\tonboard IERS EOP table file (default tests/golden/finals.all.iau2000.txt)\n"
      "-B\tonboard Chebyshev ephemeris fixture (default tests/golden/de440_bodies.cheb)\n"
      "-I\tonboard IAGA IGRF-14 coefficients (default tests/golden/igrf14coeffs.txt)\n"
      "-Y\tmission epoch for the IGRF snapshot, decimal year (default: system clock)\n"
      "-P\tParameterDb file from the config compiler (default ./PrmDb.dat)\n"
      "-M\tcommand MAG_CAL_START for N samples at startup (SITL/bench only; "
      "0/absent = no calibration)\n"
      "-A\tcommand ST_ALIGN_CAL_START as unit,pairs at startup (SITL/bench only; "
      "absent = no alignment calibration)\n",
      app);
}

/**
 * \brief shutdown topology cycling on signal
 *
 * The reference topology allows for a simulated cycling of the rate groups. This simulated cycling
 * needs to be stopped in order for the program to shutdown. This is done via handling signals such
 * that it is performed via Ctrl-C
 *
 * @param signum
 */
static void signalHandler(int signum) {
  flight::stopRateGroups();
}

/**
 * \brief execute the program
 *
 * This F´ program is designed to run in standard environments (e.g. Linux/macOs running on a
 * laptop). Thus it uses command line inputs to specify how to connect.
 *
 * @param argc: argument count supplied to program
 * @param argv: argument values supplied to program
 * @return: 0 on success, something else on failure
 */
int main(int argc, char* argv[]) {
  I32 option = 0;
  CHAR* hostname = nullptr;
  U16 port_number = 0;
  U16 sitl_port = 0;
  bool scripted_commands = false;
  const char* onboard_eop_path = nullptr;
  const char* onboard_ephem_path = nullptr;
  const char* onboard_igrf_path = nullptr;
  const char* prm_db_path = nullptr;
  double igrf_epoch_year = 0.0;  // 0 = derive from the system clock at setup
  U32 mag_cal_samples = 0;       // 0 = do not command a calibration at startup
  U8 st_align_unit = 1;          // starTrackerIn index the startup alignment names
  U32 st_align_samples = 0;      // 0 = do not command an alignment calibration at startup

  Os::init();

  // Loop while reading the getopt supplied options
  while ((option = getopt(argc, argv, "hp:a:s:cE:B:I:Y:P:M:A:")) != -1) {
    switch (option) {
      // Handle the -a argument for address/hostname
      case 'a':
        hostname = optarg;
        break;
      // Handle the -p port number argument
      case 'p':
        port_number = static_cast<U16>(atoi(optarg));
        break;
      // Handle the -s SITL lockstep port argument (design doc §2.2). Reject
      // garbage/out-of-range values outright rather than silently truncating
      // (or silently disabling SITL): a wrong port must fail loudly.
      case 's': {
        char* end = nullptr;
        const long parsed = strtol(optarg, &end, 10);
        if (end == optarg || *end != '\0' || parsed < 1 || parsed > 65535) {
          (void)printf("Invalid SITL port '%s' (expected 1-65535)\n", optarg);
          return 1;
        }
        sitl_port = static_cast<U16>(parsed);
        break;
      }
      // Enable the placeholder scripted actuator profile (SITL only, §2.4)
      case 'c':
        scripted_commands = true;
        break;
      // Onboard-table paths (design doc §11.3, §22); absent = topology defaults.
      case 'E':
        onboard_eop_path = optarg;
        break;
      case 'B':
        onboard_ephem_path = optarg;
        break;
      // Onboard IGRF-14 coefficient file (design doc §6.2, §8.1).
      case 'I':
        onboard_igrf_path = optarg;
        break;
      // Mission epoch the IGRF snapshot is taken at, as a decimal year. Which
      // snapshot the vehicle holds is a ground decision (§19.3); absent this,
      // setup falls back to the system clock.
      case 'Y': {
        char* end = nullptr;
        const double parsed = strtod(optarg, &end);
        if (end == optarg || *end != '\0' || !(parsed > 0.0)) {
          (void)printf("Invalid IGRF epoch year '%s' (expected e.g. 2026.5)\n", optarg);
          return 1;
        }
        igrf_epoch_year = parsed;
        break;
      }
      // ParameterDb file the config compiler emitted for this vehicle (§19.3).
      // An empty path is rejected here rather than passed on: PrmDb asserts on a
      // zero-length filename, so `-P ""` would abort the deployment at setup.
      case 'P':
        if (optarg[0] == '\0') {
          (void)printf("Invalid ParameterDb path: expected a filename\n");
          return 1;
        }
        prm_db_path = optarg;
        break;
      // Startup magnetometer calibration (design doc §8.1), for the SITL
      // demonstration and bench runs where no ground link is attached. The
      // component still range-checks the count against its own tuning; this only
      // rejects what cannot be a count at all.
      case 'M': {
        char* end = nullptr;
        const long parsed = strtol(optarg, &end, 10);
        if (end == optarg || *end != '\0' || parsed < 0 || parsed > 1000000) {
          (void)printf("Invalid MAG_CAL_START sample count '%s' (expected 0-1000000)\n", optarg);
          return 1;
        }
        mag_cal_samples = static_cast<U32>(parsed);
        break;
      }
      // Startup inter-star-tracker alignment calibration (design doc §8.2), the
      // twin of -M and there for the same reason. Argument is "unit,pairs"; the
      // component range-checks both against its own tuning, so this only rejects
      // what cannot be a unit or a count at all.
      case 'A': {
        char* end = nullptr;
        const long unit = strtol(optarg, &end, 10);
        if (end == optarg || *end != ',' || unit < 0 || unit > 255) {
          (void)printf("Invalid ST_ALIGN_CAL_START spec '%s' (expected unit,pairs)\n", optarg);
          return 1;
        }
        const char* count_str = end + 1;
        const long pairs = strtol(count_str, &end, 10);
        if (end == count_str || *end != '\0' || pairs < 0 || pairs > 1000000) {
          (void)printf("Invalid ST_ALIGN_CAL_START pair count in '%s' (expected 0-1000000)\n",
                       optarg);
          return 1;
        }
        st_align_unit = static_cast<U8>(unit);
        st_align_samples = static_cast<U32>(pairs);
        break;
      }
      // Cascade intended: help output
      case 'h':
      // Cascade intended: help output
      case '?':
      // Default case: output help and exit
      default:
        print_usage(argv[0]);
        return (option == 'h') ? 0 : 1;
    }
  }
  // Object for communicating state to the topology
  flight::TopologyState inputs;
  inputs.hostname = hostname;
  inputs.port = port_number;
  inputs.sitlPort = sitl_port;
  inputs.scriptedCommands = scripted_commands;
  inputs.onboardEopPath = onboard_eop_path;
  inputs.onboardEphemPath = onboard_ephem_path;
  inputs.onboardIgrfPath = onboard_igrf_path;
  inputs.igrfEpochYear = igrf_epoch_year;
  inputs.magCalSamples = mag_cal_samples;
  inputs.stAlignUnit = st_align_unit;
  inputs.stAlignSamples = st_align_samples;
  inputs.prmDbPath = prm_db_path;

  // Setup program shutdown via Ctrl-C
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);
  Fw::Logger::log("Hit Ctrl-C to quit\n");

  // Setup, cycle, and teardown topology
  flight::setupTopology(inputs);
  flight::startRateGroups(Fw::TimeInterval(1, 0));  // Program loop cycling rate groups at 1Hz
  flight::teardownTopology(inputs);
  Fw::Logger::log("Exiting...\n");
  return 0;
}
