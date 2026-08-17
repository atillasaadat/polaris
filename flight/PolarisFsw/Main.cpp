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
#include <cstdio>
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
      "-E\tonboard IERS EOP table file (default tests/golden/finals.all.iau2000.txt)\n"
      "-B\tonboard Chebyshev ephemeris fixture (default tests/golden/de440_bodies.cheb)\n"
      "-I\tonboard IAGA IGRF-14 coefficients (default tests/golden/igrf14coeffs.txt)\n"
      "-Y\tmission epoch for the IGRF snapshot, decimal year (default: system clock)\n"
      "-P\tParameterDb file from the config compiler (default ./PrmDb.dat)\n"
      "-c\tcontrol mode to latch at startup: 1=DETUMBLE, 2=POINT (SITL/bench "
      "only; 0/absent = stay IDLE)\n"
      "-q\tinertial-hold target quaternion q0,q1,q2,q3 (JPL scalar-first), for -c 2\n"
      "-M\tcommand MAG_CAL_START for N samples at startup (SITL/bench only; "
      "0/absent = no calibration)\n"
      "-A\tcommand ST_ALIGN_CAL_START as unit,pairs at startup (SITL/bench only; "
      "absent = no alignment calibration)\n"
      "-F\tdisturbance feedforward tiers as model,observer (0/1 each; SITL/bench "
      "only; absent = the ParameterDb values)\n"
      "-R\tcommand OD_RESET on GNC cycle N (SITL/bench only; 0/absent = never)\n"
      "-W\twheel-speed bias 0/1 (SITL/bench only; absent = the ParameterDb pattern)\n",
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
  U32 ctrl_mode = 0;                            // 0 = leave the controller in IDLE
  F64 ctrl_target_q[4] = {0.0, 0.0, 0.0, 0.0};  // null norm = no target commanded
  const char* onboard_eop_path = nullptr;
  const char* onboard_ephem_path = nullptr;
  const char* onboard_igrf_path = nullptr;
  const char* prm_db_path = nullptr;
  double igrf_epoch_year = 0.0;  // 0 = derive from the system clock at setup
  U32 mag_cal_samples = 0;       // 0 = do not command a calibration at startup
  U8 st_align_unit = 1;          // starTrackerIn index the startup alignment names
  U32 st_align_samples = 0;      // 0 = do not command an alignment calibration at startup
  I32 ff_model = -1;             // <0 = leave the ParameterDb value alone
  I32 ff_observer = -1;
  U32 od_reset_cycle = 0;  // 0 = never command an orbit-filter reset
  I32 wheel_bias = -1;     // <0 = leave the ParameterDb pattern alone

  Os::init();

  // Loop while reading the getopt supplied options
  while ((option = getopt(argc, argv, "hp:a:s:c:q:E:B:I:Y:P:M:A:F:R:W:")) != -1) {
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
      // SITL/bench control-mode latch and inertial-hold target (design doc §8.5).
      case 'c': {
        const long parsed = ::strtol(optarg, nullptr, 10);
        if (parsed < 0 || parsed > 2) {
          (void)printf("Invalid control mode '%s' (expected 0=IDLE, 1=DETUMBLE, 2=POINT)\n",
                       optarg);
          return 1;
        }
        ctrl_mode = static_cast<U32>(parsed);
        break;
      }
      case 'q': {
        if (::sscanf(optarg, "%lf,%lf,%lf,%lf", &ctrl_target_q[0], &ctrl_target_q[1],
                     &ctrl_target_q[2], &ctrl_target_q[3]) != 4) {
          (void)printf("Invalid target quaternion '%s' (expected q0,q1,q2,q3)\n", optarg);
          return 1;
        }
        break;
      }
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
      // SITL/bench only: force the §8.5 feedforward tiers on or off, so the same
      // vehicle can be flown with and without them and the difference measured.
      case 'F': {
        int model = 0;
        int observer = 0;
        if (::sscanf(optarg, "%d,%d", &model, &observer) != 2 || model < 0 || model > 1 ||
            observer < 0 || observer > 1) {
          (void)printf("Invalid feedforward spec '%s' (expected model,observer as 0/1)\n", optarg);
          return 1;
        }
        ff_model = model;
        ff_observer = observer;
        break;
      }
      // SITL/bench only: command the orbit filter's OD_RESET on a given GNC
      // cycle, so the reset-and-reseed path can be flown with no ground link.
      case 'R': {
        char* end = nullptr;
        const long parsed = strtol(optarg, &end, 10);
        if (end == optarg || *end != '\0' || parsed < 0 || parsed > 100000000) {
          (void)printf("Invalid OD_RESET cycle '%s' (expected 0-100000000)\n", optarg);
          return 1;
        }
        od_reset_cycle = static_cast<U32>(parsed);
        break;
      }
      // SITL/bench only: force the §8.5 wheel-speed bias on or off, the
      // zero-crossing A/B experiment's switch.
      case 'W': {
        const long parsed = ::strtol(optarg, nullptr, 10);
        if (parsed < 0 || parsed > 1) {
          (void)printf("Invalid wheel-bias spec '%s' (expected 0 or 1)\n", optarg);
          return 1;
        }
        wheel_bias = static_cast<I32>(parsed);
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
  inputs.ctrlMode = ctrl_mode;
  for (int i = 0; i < 4; ++i) {
    inputs.ctrlTargetQ[i] = ctrl_target_q[i];
  }
  inputs.onboardEopPath = onboard_eop_path;
  inputs.onboardEphemPath = onboard_ephem_path;
  inputs.onboardIgrfPath = onboard_igrf_path;
  inputs.igrfEpochYear = igrf_epoch_year;
  inputs.magCalSamples = mag_cal_samples;
  inputs.stAlignUnit = st_align_unit;
  inputs.stAlignSamples = st_align_samples;
  inputs.ffModel = ff_model;
  inputs.ffObserver = ff_observer;
  inputs.odResetCycle = od_reset_cycle;
  inputs.wheelBias = wheel_bias;
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
