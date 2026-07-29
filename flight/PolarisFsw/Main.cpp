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
      "-B\tonboard Chebyshev ephemeris fixture (default tests/golden/de440_bodies.cheb)\n",
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

  Os::init();

  // Loop while reading the getopt supplied options
  while ((option = getopt(argc, argv, "hp:a:s:cE:B:")) != -1) {
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
