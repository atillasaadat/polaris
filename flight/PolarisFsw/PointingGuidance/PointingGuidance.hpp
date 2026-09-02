// ======================================================================
// \title  PointingGuidance.hpp
// \brief  Align/constrain pointing guidance (design doc §8.4; REQ-AGN-004,
//         REQ-AGN-005, REQ-ODP-002).
//
// Owns the four operator-writable stores a pointing command names — TLE slots,
// state-vector slots, ground points and custom body vectors — and turns the
// active command into an attitude and a feedforward rate once per GNC cycle.
//
// All the geometry lives in `lib/gnc/attitude_guidance`; this component is the
// F´ shell around it: uplink validation, parameter loading, the per-cycle
// gather of orbit state and ephemeris, telemetry and the events an operator
// reads. That split is deliberate — the guidance math is then testable with no
// topology, and the component has no opinion about geometry it could get wrong
// a second way.
// ======================================================================

#ifndef PolarisFsw_PointingGuidance_HPP
#define PolarisFsw_PointingGuidance_HPP

#include "flight/PolarisFsw/PointingGuidance/PointingGuidanceComponentAc.hpp"
#include "gnc/attitude_guidance.hpp"
#include "gnc/pointing_refs.hpp"
#include "gnc/target_catalog.hpp"
#include "time/leap_seconds.hpp"

namespace flight {

class PointingGuidance final : public PointingGuidanceComponentBase {
 public:
  //! A TLE line is 69 columns, and each one crosses the command boundary in two
  //! pieces because `Fw::CmdStringArg` holds only FW_CMD_STRING_MAX_SIZE (40)
  //! characters. See LOAD_TLE in the .fpp.
  static constexpr std::size_t kTleLineColumns = 69;
  static constexpr std::size_t kTleSplitColumn = 35;

  using BodyVecKind = PointingGuidance_BodyVecKind;
  using TargetKind = PointingGuidance_TargetKind;
  using GuidanceRefusal = PointingGuidance_GuidanceRefusal;
  using TargetRefusal = PointingGuidance_TargetRefusal;

  explicit PointingGuidance(const char* compName);
  ~PointingGuidance() override = default;

 public:
  //! SITL/bench only: issue a SET_GUIDANCE at startup, with no ground link
  //! attached. Runs the command handler's body through the real command port,
  //! so the uplink path's validation is exercised rather than bypassed — a row
  //! that sets an invalid pair must be refused here exactly as it would be from
  //! the ground.
  void commandGuidanceAtStartup(U32 alignVecKind, U32 alignVecIndex, bool alignVecNegate,
                                U32 alignTgtKind, U32 alignTgtIndex, bool alignTgtNegate,
                                F64 alignTgtParam0, F64 alignTgtParam1, U32 conVecKind,
                                U32 conVecIndex, bool conVecNegate, U32 conTgtKind, U32 conTgtIndex,
                                bool conTgtNegate, F64 conTgtParam0, F64 conTgtParam1);

  //! SITL/bench only: fill a state-vector target slot at startup, the same way
  //! and for the same reason as commandGuidanceAtStartup — through the real
  //! LOAD_STATE_VECTOR handler, so a bad upload is refused here as it would be
  //! from the ground. Without this a `SAT_STATE_n` row could never be flown
  //! without a ground link, which is to say could never be flown.
  void commandStateVectorAtStartup(U32 slot, I64 epochTaiNs, const F64 posM[3], const F64 velMps[3],
                                   F64 sigmaM);

  //! The TLE twin. @p verifyChecksum is passed through rather than forced,
  //! because the committed AIAA verification element sets legitimately carry
  //! stale checksums and are exactly what a row wants to fly.
  void commandTleAtStartup(U32 slot, const char* line1, const char* line2, bool verifyChecksum);

 private:
  void run_handler(FwIndexType portNum, U32 context) override;
  void orbitStateIn_handler(FwIndexType portNum, const OrbitEstimate& estimate) override;
  void parameterUpdated(FwPrmIdType id) override;

  void SET_GUIDANCE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, BodyVecKind alignVecKind,
                               U8 alignVecIndex, bool alignVecNegate, TargetKind alignTgtKind,
                               U8 alignTgtIndex, bool alignTgtNegate, F64 alignTgtParam0,
                               F64 alignTgtParam1, BodyVecKind conVecKind, U8 conVecIndex,
                               bool conVecNegate, TargetKind conTgtKind, U8 conTgtIndex,
                               bool conTgtNegate, F64 conTgtParam0, F64 conTgtParam1) override;
  void CLEAR_GUIDANCE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;
  void LOAD_TLE_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot, const Fw::CmdStringArg& line1a,
                           const Fw::CmdStringArg& line1b, const Fw::CmdStringArg& line2a,
                           const Fw::CmdStringArg& line2b, bool verifyChecksum) override;
  void LOAD_STATE_VECTOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot, I64 epochTaiNs,
                                    F64 posXM, F64 posYM, F64 posZM, F64 velXMps, F64 velYMps,
                                    F64 velZMps, F64 sigmaM) override;
  void CLEAR_TARGET_SLOT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, bool isTle, U8 slot) override;
  void SET_GROUND_POINT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot, F64 latitudeDeg,
                                   F64 longitudeDeg, F64 heightM) override;
  void CLEAR_GROUND_POINT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot) override;
  void SET_CUSTOM_BODY_VEC_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot, F64 x, F64 y,
                                      F64 z) override;
  void CLEAR_CUSTOM_BODY_VEC_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 slot) override;

  //! Reload the mounting parameters into the body-vector table.
  void reloadMountingParameters();

  //! Assemble the per-cycle context: orbit state, Sun/Moon, Earth orientation.
  //! Returns false when a required input is missing, having set @p refusal.
  bool buildContext(polaris::time::Tai now, polaris::gnc::GuidanceContext& ctx,
                    polaris::gnc::GuidanceStatus& status);

  //! Publish an invalid target and account for the refusal.
  void publishNoTarget(polaris::time::Tai now, polaris::gnc::GuidanceStatus status);

  static GuidanceRefusal toRefusal(polaris::gnc::GuidanceStatus status);
  static TargetRefusal toRefusal(polaris::gnc::TargetStatus status);

  // ---- Stores. Fixed-size, owned here; see the class comment.
  polaris::gnc::TargetCatalog catalog_;
  polaris::gnc::BodyVectorTable body_vectors_;
  polaris::gnc::GroundPointTable ground_points_;

  // ---- The active command.
  polaris::gnc::GuidanceCommand command_{};
  bool commanded_ = false;

  //! A command latched before the parameters were available (SITL/bench: see
  //! commandGuidanceAtStartup), retried once per cycle until it can be
  //! validated. The mounting parameters arrive from PrmDb *after* topology
  //! setup, so a startup command validated once at setup would always be
  //! refused with BODY_VECTOR_UNKNOWN — not because it is wrong but because
  //! nothing had loaded yet. The same latch-and-retry AttitudeController uses
  //! for its startup mode, and for the same reason: it is what a ground
  //! operator would do.
  polaris::gnc::GuidanceCommand pending_command_{};
  bool pending_ = false;

  // ---- Latest orbit solution.
  OrbitEstimate orbit_{};
  bool orbit_fresh_ = false;

  // ---- Health accounting.
  U32 refusal_streak_ = 0;
  polaris::gnc::GuidanceStatus last_status_ = polaris::gnc::GuidanceStatus::kOk;

  // ---- Tuning, read from parameters; a missing value refuses (§19.3).
  double max_orbit_age_s_ = 0.0;
  bool configured_ = false;

  polaris::time::LeapSecondTable leap_{};
};

}  // namespace flight

#endif
