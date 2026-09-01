# lib/ — shared C++ (design doc §22.3). Flight paths obey the flight memory/
# exception rules (Golden Rule 6). Include root is lib/, so headers are included
# as "constants/...", "math/...".

# --- Physical-constants registry (header-only) ------------------------------
add_library(polaris_constants INTERFACE)
target_include_directories(polaris_constants INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(polaris::constants ALIAS polaris_constants)

# --- Random: seeded, per-source-derived PRNG streams (header-only) -----------
# Determinism mandate (§3.5): every stochastic source draws from an independent
# stream so a run is bit-reproducible from {config, seed}. Used by the sensor/
# actuator truth models and, later, the Monte-Carlo dispersions.
add_library(polaris_random INTERFACE)
target_include_directories(polaris_random INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(polaris::random ALIAS polaris_random)

# --- Math: frame tags, typed vectors, quaternion, orbit-relative frames ------
add_library(polaris_math
  math/quaternion.cpp
  math/frame_geometry.cpp)
target_include_directories(polaris_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_math
  PUBLIC Eigen3::Eigen polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::math ALIAS polaris_math)

# --- Time: TAI master clock, GPS/TT/TDB scales, leap seconds, UTC derivation --
add_library(polaris_time
  time/leap_seconds.cpp
  time/tdb.cpp
  time/utc.cpp)
target_include_directories(polaris_time PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_time
  PUBLIC polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::time ALIAS polaris_time)

# --- Ephemeris: onboard Chebyshev evaluation of Sun/Moon/planet positions -----
add_library(polaris_ephemeris
  ephemeris/chebyshev.cpp
  ephemeris/analytic_sun.cpp
  ephemeris/analytic_moon.cpp)
target_include_directories(polaris_ephemeris PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_ephemeris
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::ephemeris ALIAS polaris_ephemeris)

# --- Frames: the single transform library (ECI<->ECEF, IAU 2006/2000A) + EOP --
# Links ERFA (the BSD translation of the IAU SOFA reference implementation) for
# the reduction itself; REQ-CONV-002 requires one transform library, not ad-hoc
# rotations. ERFA is PUBLIC-free: it is an implementation detail, kept out of the
# public headers, so it links PRIVATE.
add_library(polaris_frames
  frames/eci_ecef.cpp)
target_include_directories(polaris_frames PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_frames
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_constants
  PRIVATE ERFA::erfa polaris_flags)
add_library(polaris::frames ALIAS polaris_frames)

# --- Environment: shared flight/sim environment models (IGRF-14 geomagnetic) --
# Dual-use by design: the truth sim needs the field for magnetometer truth and
# residual-dipole torque, and the FSW needs the same model onboard as the
# inertial reference for measured-vs-modelled checks and coarse attitude (§6.2).
# Flight path, so it obeys the flight memory/exception rules — including the IAGA
# coefficient reader (igrf_iaga.cpp), which the FSW needs to obtain its onboard
# snapshot; sim/world/igrf_file.cpp is now just a std::string façade over it.
add_library(polaris_environment
  environment/igrf.cpp
  environment/igrf_iaga.cpp)
target_include_directories(polaris_environment PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_environment
  PUBLIC Eigen3::Eigen polaris_math polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::environment ALIAS polaris_environment)

# --- State: canonical nav product (EstimatedState / TruthState; header-only) -
add_library(polaris_state INTERFACE)
target_include_directories(polaris_state INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_state INTERFACE
  Eigen3::Eigen polaris_math polaris_time polaris_constants)
add_library(polaris::state ALIAS polaris_state)

# --- Onboard: time/EOP/ephemeris table store (flight-safe) -------------------
# The flight-side holder the OnboardTables F´ component wraps: loads leap/EOP/
# Chebyshev tables from disk, validates, and answers point queries. Wraps the
# lib tables/evaluators; file reads are cstdio at load/reload only (design doc
# §11.3, §22; REQ-CDH-002). Flight memory/exception rules apply.
add_library(polaris_onboard
  onboard/tables.cpp)
target_include_directories(polaris_onboard PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(polaris_onboard
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_frames polaris_ephemeris polaris_constants
  PRIVATE polaris_flags)
add_library(polaris::onboard ALIAS polaris_onboard)

# --- GNC: estimation/guidance/control algorithms (flight-safe) ---------------
# Phase 4 starts here with the coarse attitude chain: the TRIAD single-frame
# initializer and the SS+MAG+IMU coarse estimator behind the Safe-mode floor
# (§8.1, §10), joined by the fine mode: Davenport's q-method (N-vector
# cold-start/re-init) and the 6-state attitude/gyro-bias MEKF, plus the
# attitude-free magnetometer hard/soft-iron calibration the on-orbit MAG_CAL
# flow runs, the always-on Earth-albedo correction of the sun vector, the shared
# redundancy-vote policy with its multi-IMU and multi-magnetometer wrappers, and
# the commanded inter-star-tracker alignment calibration (§8.2). Phase 5 adds the
# control half (§8.5): the B-dot detumble law, the quaternion-error PID and the
# L2/L-infinity reaction-wheel allocation. Phase 6 adds the orbit half (§8.3):
# the GNSS-aided orbit determination filter, which is also why this target links
# polaris_frames — it owns the ECEF/GPS -> ECI/TAI ingest conversion. Pure math
# on the canonical state — no F´ types, no I/O; the F´ AttitudeEstimator and
# AttitudeController components wrap it.
add_library(polaris_gnc
  gnc/triad.cpp
  gnc/davenport.cpp
  gnc/coarse_attitude.cpp
  gnc/mekf.cpp
  gnc/mag_calibration.cpp
  gnc/albedo_correction.cpp
  gnc/unit_voting.cpp
  gnc/imu_voting.cpp
  gnc/mag_voting.cpp
  gnc/st_alignment.cpp
  gnc/bdot.cpp
  gnc/attitude_pid.cpp
  gnc/rw_allocation.cpp
  gnc/rw_friction.cpp
  gnc/momentum.cpp
  gnc/rw_bias.cpp
  gnc/disturbance.cpp
  gnc/dipole_estimation.cpp
  gnc/geopotential.cpp
  gnc/orbit_od.cpp
  # SGP4/SDP4 and its TLE input (REQ-ODP-003). Split three ways for size: the
  # near-Earth initialisation, the propagation core, and the deep-space half
  # that 24 of the 33 official verification cases exercise.
  gnc/tle.cpp
  gnc/sgp4.cpp
  gnc/sgp4_propagate.cpp
  gnc/sgp4_deep_space.cpp)
target_include_directories(polaris_gnc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# polaris_frames is PUBLIC because the onboard OD filter (§8.3) takes EOP and
# converts every GNSS fix from ECEF/GPS to ECI/TAI through the one reduction in
# the repo (REQ-CONV-001/-002) — those types are in its public header.
target_link_libraries(polaris_gnc
  PUBLIC Eigen3::Eigen polaris_math polaris_time polaris_state polaris_constants polaris_frames
  PRIVATE polaris_flags)
add_library(polaris::gnc ALIAS polaris_gnc)

# --- SITL: plant<->FSW lockstep wire format (header-only, flight-safe) -------
# One layout shared by the truth sim's TCP server and the F´ SitlBridge
# component (design doc §2.2, §2.4) so the two processes cannot drift.
add_library(polaris_sitl INTERFACE)
target_include_directories(polaris_sitl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
add_library(polaris::sitl ALIAS polaris_sitl)
