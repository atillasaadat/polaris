module flight {

  # ----------------------------------------------------------------------
  # Shared SITL rate-group port/type definitions (design doc §2.4 steps 3-4).
  #
  # The actuator command seam between the SITL rate group's command source
  # (ScriptedCmdSource today, Phase-4 GNC control later) and the SitlBridge that
  # assembles the STEP_REPLY, plus the sim-time push from SitlBridge to the SITL
  # time provider. Kept in one interface-only module so all three components
  # share exactly one definition and cannot drift.
  # ----------------------------------------------------------------------

  @ Bound on per-type unit counts carried over the SITL rate-group ports. MUST
  @ equal polaris::sitl::kMaxUnits (wire.hpp) — SitlBridge static_asserts it.
  constant SitlMaxUnits = 8

  @ Per-wheel reaction-wheel torque commands [N*m], indexed in vehicle wheel
  @ build order; entries past the wheel count are unused (zero). Torque mode only
  @ (the Phase-4 placeholder source issues no speed-mode commands).
  array WheelTorqueSet = [SitlMaxUnits] F64

  @ One magnetorquer dipole command, body frame [A*m^2].
  array MtqDipole = [3] F64

  @ Per-rod magnetorquer dipole commands, indexed in vehicle rod build order;
  @ entries past the rod count are unused (zero).
  array MtqDipoleSet = [SitlMaxUnits] MtqDipole

  @ Reaction-wheel torque command set, source -> SitlBridge (latched for the
  @ next STEP_REPLY).
  port WheelTorqueCmd(cmds: WheelTorqueSet)

  @ Magnetorquer dipole command set, source -> SitlBridge.
  port MtqDipoleCmd(cmds: MtqDipoleSet)

  @ SitlBridge -> SITL time provider: the macro-step sim epoch (TAI ns) so the
  @ FSW clock, and every EVR/telemetry timestamp, keys off sim time in a SITL run.
  port SitlTimeSet(epochTaiNs: I64)

}
