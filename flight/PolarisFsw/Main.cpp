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
      "-c\tcontrol mode to latch at startup: 1=DETUMBLE, 2=POINT, 3=TRACK "
      "(SITL/bench only; 0/absent = stay IDLE). TRACK needs -G.\n"
      "-q\tinertial-hold target quaternion q0,q1,q2,q3 (JPL scalar-first), for -c 2\n"
      "-M\tcommand MAG_CAL_START for N samples at startup (SITL/bench only; "
      "0/absent = no calibration)\n"
      "-A\tcommand ST_ALIGN_CAL_START as unit,pairs at startup (SITL/bench only; "
      "absent = no alignment calibration)\n"
      "-F\tdisturbance feedforward tiers as model,observer (0/1 each; SITL/bench "
      "only; absent = the ParameterDb values)\n"
      "-R\tcommand OD_RESET on GNC cycle N (SITL/bench only; 0/absent = never)\n"
      "-W\twheel-speed bias 0/1 (SITL/bench only; absent = the ParameterDb pattern)\n"
      "-N\torbit filter uses the burn executor's acceleration 0/1 (SITL/bench only; "
      "absent = 1)\n"
      "-b\tcommand BURN_START as cycle,durationS,throttle (SITL/bench only; absent = "
      "no burn)\n"
      "-G\tSET_GUIDANCE at startup: 16 comma-separated fields in declaration order "
      "(SITL/bench only; absent = uncommanded)\n"
      "-V\tLOAD_STATE_VECTOR at startup: slot,epochTaiNs,pX,pY,pZ,vX,vY,vZ,sigmaM "
      "(SITL/bench only; absent = no upload)\n"
      "-L\tLOAD_TLE at startup: slot|line1|line2|verifyChecksum (SITL/bench only; "
      "absent = no upload)\n",
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
  U32 ctrl_mode = 0;  // 0 = leave the controller in IDLE
  // SITL/bench align/constrain pointing command (§8.4); absent = uncommanded,
  // which is the flight default since a real one arrives by uplink.
  bool guidance_set = false;
  unsigned guidance_fields[12] = {};
  double guidance_params[4] = {};
  // SITL/bench target-catalogue uploads (§8.4). A catalogue slot is uplinked on
  // a real vehicle, so both are off by default and a row that wants to point at
  // a satellite has to say which one.
  I32 sat_state_slot = -1;
  long long sat_state_epoch_ns = 0;
  double sat_state_pos[3] = {};
  double sat_state_vel[3] = {};
  double sat_state_sigma = 0.0;
  I32 sat_tle_slot = -1;
  char sat_tle_line1[72] = {};
  char sat_tle_line2[72] = {};
  unsigned sat_tle_verify = 1;
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
  U32 od_reset_cycle = 0;   // 0 = never command an orbit-filter reset
  I32 wheel_bias = -1;      // <0 = leave the ParameterDb pattern alone
  I32 od_accel_input = -1;  // <0 = flight behaviour (the filter uses the accel input)
  U32 burn_cycle = 0;       // 0 = never command a burn
  double burn_duration_s = 0.0;
  double burn_throttle = 0.0;

  Os::init();

  // Loop while reading the getopt supplied options
  while ((option = getopt(argc, argv, "hp:a:s:c:q:E:B:I:Y:P:M:A:F:R:W:b:N:G:V:L:")) != -1) {
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
        if (parsed < 0 || parsed > 3) {
          (void)printf(
              "Invalid control mode '%s' (expected 0=IDLE, 1=DETUMBLE, 2=POINT, 3=TRACK)\n",
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
      // SITL/bench align/constrain pointing command (design doc §8.4). Sixteen
      // comma-separated SET_GUIDANCE arguments in declaration order, so a row
      // reads the same as the uplink it stands in for:
      //   alignVecKind,alignVecIndex,alignVecNegate,
      //   alignTgtKind,alignTgtIndex,alignTgtNegate,alignTgtParam0,alignTgtParam1,
      //   conVecKind,conVecIndex,conVecNegate,
      //   conTgtKind,conTgtIndex,conTgtNegate,conTgtParam0,conTgtParam1
      case 'G': {
        unsigned av_k = 0, av_i = 0, av_n = 0, at_k = 0, at_i = 0, at_n = 0;
        unsigned cv_k = 0, cv_i = 0, cv_n = 0, ct_k = 0, ct_i = 0, ct_n = 0;
        double at_p0 = 0.0, at_p1 = 0.0, ct_p0 = 0.0, ct_p1 = 0.0;
        if (::sscanf(optarg, "%u,%u,%u,%u,%u,%u,%lf,%lf,%u,%u,%u,%u,%u,%u,%lf,%lf", &av_k, &av_i,
                     &av_n, &at_k, &at_i, &at_n, &at_p0, &at_p1, &cv_k, &cv_i, &cv_n, &ct_k, &ct_i,
                     &ct_n, &ct_p0, &ct_p1) != 16) {
          (void)printf("Invalid guidance spec '%s' (expected 16 comma-separated fields)\n", optarg);
          return 1;
        }
        guidance_set = true;
        guidance_fields[0] = av_k;
        guidance_fields[1] = av_i;
        guidance_fields[2] = av_n;
        guidance_fields[3] = at_k;
        guidance_fields[4] = at_i;
        guidance_fields[5] = at_n;
        guidance_fields[6] = cv_k;
        guidance_fields[7] = cv_i;
        guidance_fields[8] = cv_n;
        guidance_fields[9] = ct_k;
        guidance_fields[10] = ct_i;
        guidance_fields[11] = ct_n;
        guidance_params[0] = at_p0;
        guidance_params[1] = at_p1;
        guidance_params[2] = ct_p0;
        guidance_params[3] = ct_p1;
        break;
      }
      // SITL/bench state-vector upload (design doc §8.4/§8.3). Nine
      // comma-separated LOAD_STATE_VECTOR arguments in declaration order:
      //   slot,epochTaiNs,posX,posY,posZ,velX,velY,velZ,sigmaM
      case 'V': {
        unsigned slot = 0;
        if (::sscanf(optarg, "%u,%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &slot, &sat_state_epoch_ns,
                     &sat_state_pos[0], &sat_state_pos[1], &sat_state_pos[2], &sat_state_vel[0],
                     &sat_state_vel[1], &sat_state_vel[2], &sat_state_sigma) != 9) {
          (void)printf("Invalid state-vector spec '%s' (expected 9 comma-separated fields)\n",
                       optarg);
          return 1;
        }
        sat_state_slot = static_cast<I32>(slot);
        break;
      }
      // SITL/bench TLE upload (design doc §8.4/§3.1). Pipe-separated because a
      // TLE line is full of spaces and periods but never a pipe:
      //   slot|line1|line2|verifyChecksum
      // The checksum flag is spelled out rather than forced on, because the
      // committed AIAA verification element sets carry stale checksums and are
      // exactly the lines a row wants to fly.
      case 'L': {
        unsigned slot = 0;
        if (::sscanf(optarg, "%u|%71[^|]|%71[^|]|%u", &slot, sat_tle_line1, sat_tle_line2,
                     &sat_tle_verify) != 4) {
          (void)printf("Invalid TLE spec '%s' (expected slot|line1|line2|verifyChecksum)\n",
                       optarg);
          return 1;
        }
        sat_tle_slot = static_cast<I32>(slot);
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
      // SITL/bench only: arm a BURN_START for a GNC cycle (§17), the burn rows'
      // way to fire the thruster with no ground link. The executor validates the
      // duration and throttle against its own tuning; this rejects only garbage.
      case 'b': {
        long cycle = 0;
        if (::sscanf(optarg, "%ld,%lf,%lf", &cycle, &burn_duration_s, &burn_throttle) != 3 ||
            cycle < 1 || cycle > 100000000) {
          (void)printf("Invalid burn spec '%s' (expected cycle,durationS,throttle)\n", optarg);
          return 1;
        }
        burn_cycle = static_cast<U32>(cycle);
        break;
      }
      // SITL/bench only: the "blind" half of the burn-in-outage A/B — the orbit
      // filter ignores the burn executor's acceleration.
      case 'N': {
        const long parsed = ::strtol(optarg, nullptr, 10);
        if (parsed < 0 || parsed > 1) {
          (void)printf("Invalid orbit-filter accel-input spec '%s' (expected 0 or 1)\n", optarg);
          return 1;
        }
        od_accel_input = static_cast<I32>(parsed);
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
  inputs.guidanceSet = guidance_set;
  inputs.alignVecKind = guidance_fields[0];
  inputs.alignVecIndex = guidance_fields[1];
  inputs.alignVecNegate = guidance_fields[2] != 0;
  inputs.alignTgtKind = guidance_fields[3];
  inputs.alignTgtIndex = guidance_fields[4];
  inputs.alignTgtNegate = guidance_fields[5] != 0;
  inputs.alignTgtParam0 = guidance_params[0];
  inputs.alignTgtParam1 = guidance_params[1];
  inputs.conVecKind = guidance_fields[6];
  inputs.conVecIndex = guidance_fields[7];
  inputs.conVecNegate = guidance_fields[8] != 0;
  inputs.conTgtKind = guidance_fields[9];
  inputs.conTgtIndex = guidance_fields[10];
  inputs.conTgtNegate = guidance_fields[11] != 0;
  inputs.conTgtParam0 = guidance_params[2];
  inputs.conTgtParam1 = guidance_params[3];
  inputs.satStateSlot = sat_state_slot;
  inputs.satStateEpochTaiNs = static_cast<I64>(sat_state_epoch_ns);
  inputs.satStatePosM[0] = sat_state_pos[0];
  inputs.satStatePosM[1] = sat_state_pos[1];
  inputs.satStatePosM[2] = sat_state_pos[2];
  inputs.satStateVelMps[0] = sat_state_vel[0];
  inputs.satStateVelMps[1] = sat_state_vel[1];
  inputs.satStateVelMps[2] = sat_state_vel[2];
  inputs.satStateSigmaM = sat_state_sigma;
  inputs.satTleSlot = sat_tle_slot;
  inputs.satTleLine1 = sat_tle_line1;
  inputs.satTleLine2 = sat_tle_line2;
  inputs.satTleVerifyChecksum = sat_tle_verify != 0;
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
  inputs.odAccelInput = od_accel_input;
  inputs.burnStartCycle = burn_cycle;
  inputs.burnDurationS = burn_duration_s;
  inputs.burnThrottle = burn_throttle;
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
