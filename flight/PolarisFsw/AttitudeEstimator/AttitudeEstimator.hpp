// ======================================================================
// \title  AttitudeEstimator.hpp
// \brief  Attitude estimator component: coarse chain + MEKF fine mode with
//         arbitration (§8.1, §10; REQ-ADET-002, REQ-ADET-003, REQ-ADET-004)
//
// The F´ wrapper around polaris::gnc::CoarseAttitudeEstimator and
// polaris::gnc::Mekf: gathers sun, magnetometer, gyro and GNSS measurements off
// its port arrays, builds the inertial references (Sun from OnboardTables,
// geomagnetic field from the onboard IGRF-14 snapshot), runs one coarse cycle
// per rate-group call, arbitrates fine vs coarse, and publishes whichever
// solution is active plus health telemetry. No estimation math and no I/O live
// here. Flight rules — no heap after init, no exceptions, fixed-size storage,
// every return code checked.
//
// The coarse chain runs every cycle regardless of mode, so the fallback under a
// demotion is a live solution rather than one that has to re-acquire from cold.
// ======================================================================

#ifndef FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP
#define FLIGHT_POLARISFSW_ATTITUDEESTIMATOR_HPP

#include <cstdint>

#include "environment/igrf.hpp"
#include "flight/PolarisFsw/AttitudeEstimator/AttitudeEstimatorComponentAc.hpp"
#include "gnc/albedo_correction.hpp"
#include "gnc/coarse_attitude.hpp"
#include "gnc/davenport.hpp"
#include "gnc/imu_voting.hpp"
#include "gnc/mag_calibration.hpp"
#include "gnc/mag_voting.hpp"
#include "gnc/mekf.hpp"
#include "gnc/st_alignment.hpp"
#include "math/frames.hpp"
#include "math/typed_vector.hpp"
#include "state/estimated_state.hpp"
#include "time/leap_seconds.hpp"
#include "time/timescales.hpp"

namespace flight {

class AttitudeEstimator final : public AttitudeEstimatorComponentBase {
 public:
  //! Longest configurable IGRF coefficient path held (fixed, no heap).
  static constexpr FwSizeType kMaxPathLength = 256;

  //! Ceiling on MAG_CAL_START's sampleCount. The accumulator is O(1) in the
  //! window length so this costs nothing to raise; it is here so a corrupted or
  //! fat-fingered uplink cannot open a window the ground would never see close
  //! (100000 samples is ~2.8 hours at the 10 Hz GNC rate). MAG_CAL_ABORT closes
  //! a window early regardless.
  static constexpr U32 kMaxCalSamples = 100000;

  //! Cycles a collection window is allowed per sample it asked for, before it
  //! closes itself and reports. Ten means a window collecting at a tenth of the
  //! cycle rate still completes; anything slower is an outage, not a
  //! collection. See @ref mag_cal_deadline_cycles_.
  static constexpr U32 kMaxCalStallFactor = 10;

  //! Residual monitors, indexed as `ResidualMonitor` (SUN, MAGNETOMETER,
  //! SUN_CROSS_UNIT). Fixed so the streak/alert state is a plain array rather
  //! than three near-identical members.
  static constexpr int kMonitorCount = 3;

  //! Construct AttitudeEstimator object
  explicit AttitudeEstimator(const char* const compName);

  //! Destroy AttitudeEstimator object
  ~AttitudeEstimator();

  //! Load the onboard IGRF-14 snapshot from the verbatim IAGA coefficient file
  //! at @p igrfPath, for mission epoch @p decimalYear (called at topology setup,
  //! before the rate groups start).
  //!
  //! @p decimalYear selects the bracketing IAGA epoch pair the snapshot is
  //! collapsed from; the secular variation is then applied per cycle at that
  //! cycle's own epoch, so any value inside the same bracket yields identical
  //! field evaluations. It is deliberately **not** read from the master clock:
  //! at setup that clock has not been served a sim epoch yet (and on a cold
  //! hardware boot the RTC may not be set), which would silently snapshot the
  //! 1970 bracket and extrapolate it half a century. Which snapshot to hold is a
  //! ground decision, like the snapshot itself — see @ref kMaxIgrfEpochGapYears
  //! for what happens if the vehicle outlives it.
  //!
  //! Emits IgrfLoaded or IgrfLoadFailed. Returns true on success; on failure the
  //! estimator still runs and still publishes body rate (Safe-mode rate damping
  //! needs it) but can never form the magnetic pair, so it cannot acquire
  //! attitude.
  bool configureIgrf(const char* igrfPath, double decimalYear);

  //! Dispatch MAG_CAL_START for @p sampleCount samples through this component's
  //! own command port (no-op when zero). Called at topology setup, *after*
  //! loadParameters(), for the SITL demonstration of the §8.1 commanded flow and
  //! for bench runs — cases where the calibration has to be commanded with no
  //! ground link attached. It is the real opcode through the real command
  //! handler; the only thing skipped is the uplink. A flight vehicle commands
  //! this from the ground and leaves the topology hook at zero.
  void commandMagCalAtStartup(U32 sampleCount);

  //! Dispatch ST_ALIGN_CAL_START on @p unit for @p sampleCount pairs through this
  //! component's own command port (no-op when zero). The star-tracker twin of
  //! @ref commandMagCalAtStartup, and it exists for the same reason: the SITL
  //! demonstration of the §8.2 commanded flow needs a command to arrive with no
  //! ground link attached. Real opcode, real handler; only the uplink is skipped.
  //!
  //! Like its twin it is safe to call at topology setup: ST_ALIGN_CAL_START reads
  //! the king index and the per-unit boresights out of ParameterDb itself rather
  //! than from the per-cycle cache, so it does not need the rate group to have run
  //! once. (It must still follow loadParameters(), for the same reason the
  //! magnetometer hook does.)
  void commandStAlignCalAtStartup(U8 unit, U32 sampleCount);

 private:
  // ----------------------------------------------------------------------
  // Handler implementations for typed input ports
  // ----------------------------------------------------------------------

  //! Rate-group entry: run one estimation cycle.
  void run_handler(FwIndexType portNum, U32 context) override;

  //! Latch one IMU's increments for the next cycle.
  void imuIn_handler(FwIndexType portNum, const ImuMeas& meas) override;

  //! Latch one sun sensor's unit vector for the next cycle.
  void sunSensorIn_handler(FwIndexType portNum, const SunSensorMeas& meas) override;

  //! Latch one magnetometer's field measurement for the next cycle.
  void magnetometerIn_handler(FwIndexType portNum, const MagnetometerMeas& meas) override;

  //! Latch one GNSS fix for the next cycle (position only is consumed).
  void gnssIn_handler(FwIndexType portNum, const GnssMeas& meas) override;

  //! Latch one star tracker's attitude solution for the next cycle (§8.2). The
  //! coarse chain never reads it — it must stay tracker-independent to remain the
  //! Safe-mode floor — but the MEKF's finest rung is built from it.
  void starTrackerIn_handler(FwIndexType portNum, const StarTrackerMeas& meas) override;

  // ----------------------------------------------------------------------
  // Command and parameter handlers
  // ----------------------------------------------------------------------

  //! RESET_ESTIMATOR: drop both solutions and re-acquire from cold — the next
  //! TRIAD for coarse, a fresh Davenport seed for fine.
  void RESET_ESTIMATOR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! MAG_CAL_START: open a calibration collection window of @p sampleCount
  //! accepted samples. Refuses (EXECUTION_ERROR + MagCalRejected) on missing
  //! tuning or an out-of-range count; the estimator is unaffected either way.
  void MAG_CAL_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U32 sampleCount) override;

  //! MAG_CAL_ABORT: close a window without fitting, keeping any applied
  //! calibration. Idempotent.
  void MAG_CAL_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! MAG_CAL_CLEAR: drop the applied calibration; consumers revert to raw. Does
  //! not touch a window in progress. Idempotent.
  void MAG_CAL_CLEAR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! ST_ALIGN_CAL_START: open an inter-tracker alignment window on @p unit
  //! against the king. Refuses (EXECUTION_ERROR + StAlignRejected) on missing
  //! tuning, a bad unit index, or an out-of-range count; the estimator is
  //! unaffected either way.
  void ST_ALIGN_CAL_START_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 unit,
                                     U32 sampleCount) override;

  //! ST_ALIGN_CAL_ABORT: close a window without fitting, keeping any applied
  //! alignment. Idempotent.
  void ST_ALIGN_CAL_ABORT_cmdHandler(FwOpcodeType opCode, U32 cmdSeq) override;

  //! ST_ALIGN_CAL_CLEAR: drop the applied alignment for @p unit; that tracker
  //! reverts to as-mounted. Does not touch a window in progress. Idempotent.
  void ST_ALIGN_CAL_CLEAR_cmdHandler(FwOpcodeType opCode, U32 cmdSeq, U8 unit) override;

  //! A parameter changed: re-read the whole set on the next cycle.
  void parameterUpdated(FwPrmIdType id) override;

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  //! Read the coarse-chain parameters from ParameterDb into a config and
  //! rebuild the estimator with it. Emits ConfigInvalid (edge-gated) and leaves
  //! the estimator unconfigured if any parameter is missing or out of range —
  //! there are no flight defaults (§19.3). Returns true when configured.
  bool refreshCoarseConfig();

  //! Read the fine-mode parameters and rebuild the MEKF with them. Independent
  //! of refreshCoarseConfig(): a missing fine parameter costs the fine mode
  //! (FineConfigInvalid, coarse-only operation), not the whole estimator.
  //! Returns true when the filter is configured.
  bool refreshFineConfig();

  //! Read the three Earth-albedo-correction parameters and rebuild the
  //! correction config. Independent of the other two refreshes again: a missing
  //! albedo parameter costs the *correction* (AlbedoConfigInvalid, every cycle
  //! weighted at the uncorrected sigma), not a mode. Returns true when the
  //! correction is configured.
  bool refreshAlbedoConfig();

  //! Emit ConfigInvalid(@p detail) if not already flagged, and leave the
  //! estimator inert. Never substitutes a value (§19.3).
  void failConfig(const char* detail);

  //! Emit AlbedoConfigInvalid(@p detail) if not already flagged and leave the
  //! correction inert. Neither estimator is touched — they run uncorrected.
  void failAlbedoConfig(const char* detail);

  //! Read the star-tracker fusion parameters (king unit, the two sigmas, the
  //! per-unit boresights, the three residual-monitor thresholds and their
  //! persistence). A **fourth** independent gate: a missing value costs tracker
  //! fusion — StConfigInvalid, the ladder capped at SS+MAG — not the fine mode
  //! and not the estimator. Returns true when tracker fusion is configured.
  bool refreshStConfig();

  //! Emit StConfigInvalid(@p detail) if not already flagged and leave tracker
  //! fusion inert. Neither estimator is touched; the ladder simply cannot reach
  //! its top rung.
  void failStConfig(const char* detail);

  //! Apply the Earth-albedo correction to @p sunBody in place, and set this
  //! cycle's sun sigmas (@ref sigma_sun_sys_cycle_, @ref sigma_sun_total_cycle_)
  //! to match what actually happened. The one application point: called between
  //! unit selection and every consumer.
  //!
  //! @param sunIndex port index of the selected unit, which is what picks its
  //!        boresight out of the per-unit parameter (§8.2). A slot written as
  //!        the zero vector takes the uncorrected path.
  //! @param sunBody [in,out] the measured sun direction, replaced by the
  //!        corrected one on success and left untouched otherwise.
  //! @return the pull angle removed [rad], or NaN when the correction did not
  //!         run — which is a normal, frequent condition and leaves the cycle on
  //!         the uncorrected sigma.
  //! Compose this cycle's sun systematic and its MEKF-inflated total from the
  //! sensor-side albedo term and the reference-side ephemeris term, which are
  //! independent and therefore add in quadrature.
  void setSunSigmaForCycle(double albedoSigmaRad, double ephemSigmaRad);

  //! TODO: eight parameters is one past comfortable. Fold the geometry into a
  //! small input struct **when a third reference term arrives** — not before,
  //! since a struct for one call site is indirection without a payer. (The §8.2
  //! per-unit boresights did not trigger it: they replaced the measurement
  //! pointer with an index, leaving the count where it was.)
  double applyAlbedoCorrection(
      FwIndexType sunIndex, const polaris::math::Vec3<polaris::math::frames::ECEF>& r_ecef,
      const polaris::math::Quat<polaris::math::frames::ECI, polaris::math::frames::ECEF>&
          q_eci_ecef,
      const polaris::math::Vec3<polaris::math::frames::ECI>& sun_geocentric,
      bool havePositionAndRotation, bool haveSunGeocentric, double ephemSigmaRad,
      polaris::math::Vec3<polaris::math::frames::Body>& sunBody);

  //! Read the seven MagCal* parameters and rebuild the calibration accumulator.
  //! Called from MAG_CAL_START rather than per cycle: a missing calibration
  //! parameter costs only the ability to *start* a calibration, so it is a
  //! command-time refusal (MagCalRejected(CONFIG)) and never a flight event.
  //! Returns true when the accumulator is configured.
  bool refreshMagCalConfig();

  //! Feed one raw magnetometer reading and its modelled field magnitude to an
  //! open collection window, and fit when the target is reached. No-op unless a
  //! window is open. @p m_raw is the **uncorrected** reading — the fit must
  //! never see its own correction — and @p igrf_magnitude_t the onboard IGRF
  //! magnitude at this cycle's position and epoch.
  void collectMagSample(const polaris::math::Vec3<polaris::math::frames::Body>& m_raw,
                        double igrf_magnitude_t);

  //! Close the open window and try the fit: MagCalComplete + apply on success,
  //! MagCalRejected + nothing applied on refusal. No-op unless a window is open.
  void finishMagCal();

  //! Drop the applied calibration and emit MagCalCleared, if one is applied.
  //! Shared by MAG_CAL_CLEAR and the full RESET_ESTIMATOR semantics.
  void clearMagCal();

  //! Emit FineConfigInvalid(@p detail) if not already flagged, demote fine mode
  //! if it was engaged, and leave the MEKF inert. The coarse chain is untouched.
  void failFineConfig(const char* detail);

  //! Combine every IMU on the port array into one body rate (§8.2): per-unit
  //! plausibility gates, then a per-axis median at three or more surviving
  //! units. Never an average — one railed unit would drag it without limit
  //! (gnc/imu_voting.hpp). Emits the exclusion/re-admission FDIR events and
  //! writes ImuContributing / ImuExclusionMask.
  //!
  //! @param nowTaiNs cycle epoch, for the §9.1 staleness gate.
  //! @param rate [out] the voted body rate [rad/s], written only on success.
  //! @return true when the vote produced a usable rate.
  bool voteImuRate(I64 nowTaiNs, polaris::math::Vec3<polaris::math::frames::Body>& rate);

  //! **Best-illuminated** valid, fresh sun sensor: of the units reporting the
  //! sun in view with a usable sigma, the one with the smallest realised
  //! `sigmaRad` (§6.4) — which is the incidence-cosine criterion expressed in
  //! the quantity the estimator consumes. Ties break to the lowest index, so the
  //! choice is deterministic. Returns nullptr when nothing is selectable, and
  //! writes the chosen port index to @p index (unchanged when none).
  //!
  //! @param runnerUp receives the second-best unit's port index, or -1 when only
  //!        one unit sees the Sun. That is the input to the §8.2 cross-unit
  //!        consistency check: the selector orders on *reported* σ, so a unit that
  //!        is confidently wrong wins, and nothing else would notice.
  const SunSensorMeas* selectSunSensor(I64 nowTaiNs, FwIndexType& index,
                                       FwIndexType& runnerUp) const;

  //! Cross-check the selected sun sensor against the runner-up (§8.2), and choose
  //! which of the two to use. Writes `SunCrossUnitRad`, runs the SUN_CROSS_UNIT
  //! residual monitor, and — only once that monitor has alerted, and only with a
  //! usable fine solution to judge against — switches to the runner-up when it
  //! agrees better, emitting SunUnitOverridden.
  //!
  //! Needs no attitude of its own, which is what makes it the one cross-check
  //! available in Safe mode; the attitude is needed only to *resolve* a
  //! disagreement, not to detect one.
  //!
  //! @param index [in,out] the selected unit, replaced by the runner-up on an
  //!        override.
  //! @param runnerUp the second-best unit, or -1.
  //! @param reference the fine solution's predicted sun direction in body axes.
  //! @param haveReference @p reference is usable.
  //! @return the unit to use (may be the runner-up), or nullptr when @p index is
  //!         out of range.
  const SunSensorMeas* crossCheckSunUnit(
      FwIndexType& index, FwIndexType runnerUp,
      const polaris::math::Vec3<polaris::math::frames::Body>& reference, bool haveReference);

  //! Combine every magnetometer on the port array into one raw field (§8.2):
  //! per-unit magnitude gates against the modelled IGRF magnitude, then a per-axis
  //! median at three or more surviving units, or pairwise detection with the
  //! rotated modelled field attributing a disagreement at exactly two. Never an
  //! average (gnc/mag_voting.hpp). Emits the exclusion/re-admission FDIR edges and
  //! writes MagContributing / MagExclusionMask / MagUnitSelected.
  //!
  //! @param nowTaiNs cycle epoch, for the §9.1 staleness gate.
  //! @param magRefBody the modelled field rotated into body axes by the published
  //!        attitude — the identification reference.
  //! @param haveAttitude an attitude was available to rotate it with.
  //! @param attitudeSigmaRad that attitude's per-axis 1σ [rad], which is the
  //!        quality gate on the reference.
  //! @param modelledMagnitudeT `‖B_IGRF‖` [T] at this cycle's position.
  //! @param field [out] the voted **raw** field [T], written only on success.
  //! @param index [out] the port index whose reading was published.
  //! @return true when the vote produced a usable field.
  bool voteMagField(I64 nowTaiNs,
                    const polaris::math::Vec3<polaris::math::frames::Body>& magRefBody,
                    bool haveAttitude, double attitudeSigmaRad, double modelledMagnitudeT,
                    polaris::math::Vec3<polaris::math::frames::Body>& field, FwIndexType& index);

  //! First valid, fresh unit on its port array, or nullptr. "Fresh" is
  //! |now - timeTag| <= MaxMeasAgeSec (§9.1 staleness gate). Bounded loop; index
  //! order is vehicle build order, so this is a deterministic priority. Kept as
  //! first-valid honestly: there is one GNSS receiver, and a combination rule with
  //! no redundancy to exercise is untested code.
  const GnssMeas* selectGnss(I64 nowTaiNs) const;

  //! One star tracker's contribution to a cycle: its solution already stated in
  //! the king's frame, and the body-axes measurement covariance built from its own
  //! boresight. Fixed-size and passed by array — no heap on the 10 Hz path.
  struct StarTrackerSample {
    //! Body ← ECI, through the unit's nominal mounting and (for a non-king unit)
    //! its fitted inter-tracker alignment.
    polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI> attitude{};
    //! `R = σ_xy²(I − b bᵀ) + σ_z² b bᵀ` [rad²], body axes. The anisotropy is the
    //! whole reason two non-parallel trackers beat two parallel ones.
    Eigen::Matrix3d noise_cov{Eigen::Matrix3d::Identity()};
    //! Source port index, for telemetry and the alignment tap.
    FwIndexType index{0};
  };

  //! Gather this cycle's usable star-tracker solutions: valid, fresh, with a
  //! configured boresight, corrected into the king's frame, each with its `R`.
  //!
  //! @param nowTaiNs cycle epoch, for the §9.1 staleness gate.
  //! @param out receives up to `NUM_STARTRACKERIN_INPUT_PORTS` samples, king first
  //!        so the filter sees the frame-defining unit before any unit stated
  //!        relative to it — which matters only for determinism, since the update
  //!        is order-independent to first order, but determinism is worth having.
  //! @param validMask [out] bit i set for each unit that produced a sample.
  //! @return the number of samples written; 0 when tracker fusion is unconfigured.
  int collectStarTrackers(I64 nowTaiNs, StarTrackerSample* out, U32& validMask) const;

  //! Fold this cycle's tracker samples into the MEKF as attitude measurements.
  //!
  //! @param samples from @ref collectStarTrackers.
  //! @param count number of samples.
  //! @param accepted [out] at least one update was applied — which is what puts
  //!        the ladder on its top rung. **Overwritten**, not accumulated.
  //! @param refused [in,out] **accumulated**: set true if the filter refused an
  //!        update (malformed input), left alone otherwise, so the caller's
  //!        propagate-refusal survives this call. Feeds REFUSAL_STREAK.
  //! @param nisRejected [in,out] **accumulated** on the same contract: set true if
  //!        the χ²₃ gate rejected an update. Per-unit NIS streaks are maintained
  //!        inside; this out-parameter is only the cycle's "something was
  //!        rejected" flag.
  //! @return the largest NIS of the cycle's tracker updates, or NaN if none ran.
  double fuseStarTrackers(const StarTrackerSample* samples, int count, bool& accepted,
                          bool& refused, bool& nisRejected);

  //! Decide what a **persistently gate-rejected** star tracker means while the
  //! fine solution is *not* tracker-sourced, and act on it (§8.2, §9.2;
  //! REQ-FDIR-013).
  //!
  //! The rejection carries no information about the tracker on its own. A
  //! SS+MAG solution's error is dominated by the magnetometer's systematic,
  //! which the filter's white-`R` model averages its covariance down through, so
  //! `S = HPHᵀ + R` shrinks to milliradians while the true error stays tens —
  //! and *every* arriving tracker update fails the χ² gate however good the unit
  //! is. Latching the unit out there blames the better instrument for the
  //! filter's overconfidence, and permanently: the re-admission comparison runs
  //! against the same solution.
  //!
  //! So the verdict is taken against the **coarse** solution instead, whose
  //! covariance converges to the systematic floor rather than to zero (§8.1) and
  //! therefore does not lie, in the **Mahalanobis** metric (the coarse covariance
  //! is strongly anisotropic — roll about the sun line grows as `1/sin²θ` — so an
  //! isotropic radius is too tight across it and too loose along it at once):
  //!  - `d² ≤ χ²₃(0.999)` — the tracker is consistent with everything the vector
  //!    data supports, so the filter is the outlier and is **re-seeded from the
  //!    tracker** (FineReseededFromStarTracker);
  //!  - above it — the unit is not adopted this cycle and the refusal is reported
  //!    at a bounded cadence (FineTrackerAdoptionRefused). **Nothing is latched**:
  //!    a conviction on coarse agreement whose parole test is *fine* agreement
  //!    (@ref readmitStarTrackers) is permanent by construction, so the unit stays
  //!    a candidate and a later cycle with a better reference can still take it.
  //!
  //! No coarse solution to judge against — invalid, or a covariance that is not
  //! positive-definite — means no verdict: neither action fires. The refusal
  //! additionally needs a **fresh** coarse fix, because coast growth is a stated
  //! lower bound on the true uncertainty and an accusation must not rest on a
  //! number known to be optimistic; adoption does not, since it errs the
  //! permissive way.
  //!
  //! @param epoch this cycle's epoch, for the re-seed.
  //! @param coarse this cycle's coarse product — the honest reference.
  //! @param samples this cycle's tracker samples, @p count of them.
  //! @return true when the filter was re-seeded, which makes the cycle a
  //!         tracker-sourced one and suppresses the vector updates below.
  bool arbitrateRejectedTrackers(const polaris::time::Tai& epoch,
                                 const polaris::gnc::CoarseAttitudeOutput& coarse,
                                 const StarTrackerSample* samples, int count);

  //! Run the §8.2 residual monitors on the sources the fine solution is **not**
  //! using this cycle, and write their telemetry.
  //!
  //! Called **every** cycle rather than only while a tracker is fused, with
  //! @p active false on the cycles where the sun and magnetic pairs are
  //! measurements rather than monitors. Monitoring a source the filter is already
  //! folding in would be circular — the residual is small because the update made
  //! it small — so those cycles report NaN and reset the streaks. One call site
  //! is what keeps the NaN and the reset from being forgotten on one path.
  void updateResidualMonitors(
      const polaris::gnc::CoarseAttitudeInput& in,
      const polaris::math::Quat<polaris::math::frames::Body, polaris::math::frames::ECI>& fine,
      bool active);

  //! Advance one monitor's streak against @p threshold and emit its alert /
  //! recovery edges. Shared by all three so the persistence rule cannot drift
  //! between them.
  //!
  //! @param monitor which monitor, indexing @ref monitor_streak_.
  //! @param residualRad the measured residual, or NaN when it could not be
  //!        computed — which resets the streak without clearing an alert, because
  //!        "we stopped looking" is not "it recovered".
  void noteMonitorResidual(ResidualMonitor::T monitor, double residualRad, double thresholdRad);

  //! Advance the probation of star trackers latched out by the per-unit NIS
  //! policy, and re-admit one that has agreed with the fine solution for
  //! `MonitorAlertCycles` consecutive cycles. Judged on the criterion that
  //! excluded it — agreement with the filter — since an excluded unit is not
  //! fused and so cannot earn credit by being accepted. No-op without a valid
  //! fine solution to judge against.
  void readmitStarTrackers(I64 nowTaiNs);

  //! Read the three StAlign* parameters and rebuild the alignment accumulator.
  //! Called from ST_ALIGN_CAL_START rather than per cycle, on the same reasoning
  //! as the magnetometer calibration: a missing value costs the ability to *start*
  //! a calibration, so it is a command-time refusal and never a flight event.
  bool refreshStAlignConfig();

  //! Feed one cycle's tracker readings to an open alignment window. No-op unless a
  //! window is open, or unless both the king and the commanded unit produced a
  //! **simultaneous** valid solution this cycle. The readings passed here are
  //! **uncorrected** — a fit fed its own correction would refit the identity.
  void collectStAlignSample(I64 nowTaiNs);

  //! Close the open alignment window and try the fit: StAlignComplete + apply on
  //! success, StAlignRejected + nothing applied on refusal. No-op unless a window
  //! is open.
  void finishStAlign();

  //! Drop the applied alignment for @p unit and emit StAlignCleared, if one is
  //! applied. Shared by ST_ALIGN_CAL_CLEAR and the full RESET_ESTIMATOR semantics.
  void clearStAlign(FwIndexType unit);

  //! Unit @p index's star-tracker boresight from the flattened per-unit parameter,
  //! or false when that slot is the zero vector (not installed, or an
  //! uncharacterised mounting) — in which case that unit is not fused at all,
  //! since its measurement covariance cannot be built without a boresight.
  bool stBoresightFor(FwIndexType index,
                      polaris::math::Vec3<polaris::math::frames::Body>& out) const;

  //! Unit @p index's albedo boresight from the flattened per-unit parameter, or
  //! false when that slot is the zero vector (not installed, or an
  //! uncharacterised mounting) — in which case the correction is skipped for
  //! that unit rather than run on a guessed direction.
  bool sunBoresightFor(FwIndexType index,
                       polaris::math::Vec3<polaris::math::frames::Body>& out) const;

  //! One cycle of fine-mode arbitration, after the coarse cycle has run and
  //! with the same measurement set. Engages, steps, or demotes fine mode; never
  //! touches the coarse solution. @p coarse is the cycle's coarse product, whose
  //! validity gates a promotion. Returns the cycle's NIS for telemetry, or NaN
  //! when no update ran — so the channel is written exactly once per cycle.
  //! @param stars this cycle's usable star-tracker samples, and @p starCount how
  //!        many. They are what decides the ladder's rung: with at least one
  //!        accepted, the sun and magnetic pairs are not fused at all.
  double arbitrateFineMode(const polaris::time::Tai& epoch,
                           const polaris::gnc::CoarseAttitudeInput& in,
                           const polaris::gnc::CoarseAttitudeOutput& coarse,
                           const StarTrackerSample* stars, int starCount);

  //! Propagate and update the engaged filter on this cycle's measurements,
  //! maintaining the refusal and NIS-rejection streaks and demoting when a
  //! streak, the fine coast horizon, or an internal fault says to. Returns the
  //! largest NIS seen this cycle (NaN if no update ran).
  //!
  //! **The ladder is applied here.** Tracker updates go in first; if any is
  //! accepted the sun and magnetic pairs are skipped as measurements and handed to
  //! @ref updateResidualMonitors instead.
  //! @param coarse this cycle's coarse product, which @ref arbitrateRejectedTrackers
  //!        needs as the one attitude reference whose covariance is honest.
  double stepFineMode(const polaris::time::Tai& epoch, const polaris::gnc::CoarseAttitudeInput& in,
                      const polaris::gnc::CoarseAttitudeOutput& coarse,
                      const StarTrackerSample* stars, int starCount);

  //! Try to seed the MEKF: from a star tracker's own solution and covariance when
  //! one is available, otherwise from a Davenport solve over this cycle's vector
  //! pairs. Emits FineModeEngaged on success, edge-gated FineInitFailed on
  //! refusal.
  //!
  //! A tracker seed needs no sun/field geometry and no coarse solution
  //! underneath it, which is what makes the top rung reachable in eclipse and at
  //! cold start — the case a Davenport seed structurally cannot cover.
  void tryPromoteFineMode(const polaris::time::Tai& epoch,
                          const polaris::gnc::CoarseAttitudeInput& in,
                          const StarTrackerSample* stars, int starCount);

  //! Give up the fine solution for @p reason: emit FineModeDemoted, drop the
  //! filter state (a solution no longer trusted is not worth carrying), and
  //! clear the streaks. No-op when fine mode is not engaged.
  void demoteFineMode(FineDemotionReason::T reason);

  //! Publish the coarse product on estimateOut (if connected) and write the
  //! solution telemetry channels, stamping the product with @p epoch.
  void publishCoarse(const polaris::gnc::CoarseAttitudeOutput& out,
                     const polaris::time::Tai& epoch);

  //! Publish the fine (MEKF) product the same way.
  void publishFine(const polaris::time::Tai& epoch);

  //! Fill and emit the AttitudeEstimate port struct and the solution telemetry
  //! from `state_`, which the caller has already written from its estimator.
  //! @p cov is the attitude-error covariance to report [rad²] and @p age_s the
  //! solution age; both belong to the active solution.
  void emitEstimate(const Eigen::Matrix3d& cov, double age_s);

  //! Emit ReferenceDegraded/ReferenceRecovered for @p domain on a transition of
  //! its served grade, then latch @p grade into @p last. Per-domain so the alert
  //! names which reference degraded, following the OnboardTables pattern.
  void noteReferenceGrade(TableDomain::T domain, TableGrade::T grade, TableGrade::T& last,
                          bool& have);

  //! Emit the AttitudeLost edge if a valid solution is being given up, and clear
  //! the latch. Called from every path that drops the solution — coast expiry,
  //! configuration failure, and a tuning change that rebuilds the estimator —
  //! so a consumer sees the same edge however the attitude went away.
  void noteAttitudeLost();

  //! Current master-clock epoch as TAI nanoseconds.
  I64 currentTaiNs();

  // ----------------------------------------------------------------------
  // State
  // ----------------------------------------------------------------------

  //! The estimator proper. Starts inert (default config fails validation) and is
  //! rebuilt by refreshConfig() once every parameter is present and in range.
  polaris::gnc::CoarseAttitudeEstimator estimator_;

  //! The fine estimator. Starts inert on the same rule, and is rebuilt by
  //! refreshFineConfig(); an inert filter simply means coarse-only operation.
  polaris::gnc::Mekf mekf_;

  //! Fault-tolerant combiner for the redundant IMU set (§8.2). Holds the
  //! exclusion latch and re-admission counters across cycles, so it is component
  //! state rather than a per-cycle call. Starts inert on the same no-defaults
  //! rule as the estimators; refreshCoarseConfig() builds it, because a vehicle
  //! with no voted rate has no estimator either.
  polaris::gnc::ImuVoter imu_voter_;

  //! Fault-tolerant combiner for the redundant magnetometer set (§8.2). Same
  //! shape, same lifetime and same gate as @ref imu_voter_: it holds an exclusion
  //! latch across cycles, and refreshCoarseConfig() builds it, because a vehicle
  //! with no voted field has no TRIAD and therefore no coarse attitude.
  polaris::gnc::MagVoter mag_voter_;

  //! Streaming quaternion-average accumulator for the commanded inter-tracker
  //! alignment (§8.2). One 4×4 moment matrix and a count — O(1) in the window
  //! length, so a collection window allocates nothing. Inert until
  //! refreshStAlignConfig() builds it at ST_ALIGN_CAL_START.
  polaris::gnc::StAlignmentAccumulator st_align_accumulator_;

  //! Streaming ellipsoid accumulator for the commanded calibration (§8.1).
  //! Fixed storage — a 10x10 normal matrix and a handful of moments, O(1) in the
  //! window length — so a collection window allocates nothing. Starts inert on
  //! the same rule as the estimators; refreshMagCalConfig() builds it.
  polaris::gnc::MagCalibrationAccumulator mag_cal_accumulator_;

  //! The applied calibration. Default-constructed means `valid == false`, and
  //! applyMagCalibration() then passes the raw reading through unchanged — which
  //! is why the magnetometer path calls it unconditionally.
  polaris::gnc::MagCalibrationResult mag_cal_{};

  //! Onboard IGRF-14 snapshot. Inert until configureIgrf() succeeds.
  polaris::environment::IgrfField igrf_;

  //! Base epoch [decimal years] of the loaded snapshot (telemetry/EVR context).
  double igrf_epoch_year_{0.0};

  //! Decimal year past which the loaded snapshot stops being the published
  //! model; the magnetic reference is refused beyond it.
  double igrf_valid_until_year_{0.0};

  //! In-code IERS leap-second record, for the TAI->decimal-year reduction the
  //! field model is parameterised by. Independent of any upload, like the
  //! OnboardTables ΔAT path (Push 38), so the coarse chain stays table-independent.
  polaris::time::LeapSecondTable leap_;

  //! Canonical onboard state (§8.0) written by gnc::writeToEstimatedState each
  //! cycle; the published port struct is filled from it.
  polaris::state::EstimatedState state_{};

  //! Latched measurements, one slot per port. Default-constructed slots have
  //! `valid == false`, so a never-connected unit is simply never selected.
  ImuMeas imu_[NUM_IMUIN_INPUT_PORTS]{};
  SunSensorMeas sun_[NUM_SUNSENSORIN_INPUT_PORTS]{};
  MagnetometerMeas mag_[NUM_MAGNETOMETERIN_INPUT_PORTS]{};
  GnssMeas gnss_[NUM_GNSSIN_INPUT_PORTS]{};
  StarTrackerMeas star_[NUM_STARTRACKERIN_INPUT_PORTS]{};

  //! Applied inter-tracker alignment, per unit. Default-constructed means
  //! `valid == false`, and `applyStAlignment` then passes the reading through
  //! unchanged — which is why the tracker path calls it unconditionally, and why
  //! the king's slot is simply never filled.
  polaris::gnc::StAlignmentResult st_align_[NUM_STARTRACKERIN_INPUT_PORTS]{};

  char igrf_path_[kMaxPathLength]{};

  //! Staleness gate [s], cached from the parameter set with the rest of the config.
  F64 max_meas_age_s_{0.0};

  //! §9.1 range gate on a GNSS fix [m], geocentric radius. Zero until configured,
  //! which rejects every fix — the estimator is inert without tuning anyway.
  F64 min_position_radius_m_{0.0};
  F64 max_position_radius_m_{0.0};

  //! Magnetic 1-sigma handed to the MEKF and the Davenport seed [rad]: the white
  //! and systematic parts of the same budget, root-sum-squared. The filter treats
  //! R as white, so the systematic part has to be inflated into sigma or the
  //! covariance it reports converges below the true error (gnc/mekf.hpp). The sun
  //! side has no equivalent constant — it is per-cycle, below.
  F64 sigma_mag_total_rad_{0.0};

  //! The four terms this cycle's sun systematic is composed from, cached from
  //! the parameter set so the per-cycle choice is two comparisons and a hypot
  //! rather than a parameter read. Two are the **sensor's** (albedo, with and
  //! without the correction) and two the **reference's** (ephemeris, tables and
  //! analytic fallback); which of each is in force is decided per cycle by
  //! whether the correction ran and by the served ephemeris grade.
  F64 sigma_sun_white_rad_{0.0};
  F64 sigma_sun_albedo_corr_rad_{0.0};
  F64 sigma_sun_albedo_uncorr_rad_{0.0};
  F64 sigma_sun_ephem_rad_{0.0};
  F64 sigma_sun_ephem_precise_rad_{0.0};

  //! **This cycle's** sun-pair systematic σ and its MEKF-inflated total [rad],
  //! written by setSunSigmaForCycle() before any consumer reads them. Not a
  //! choice between two constants: on a corrected cycle the albedo term carries
  //! the attitude-error-driven `A·σ_att/2`, which depends on how well the
  //! attitude is known *now*, so a freshly acquired 10° solution is weighted
  //! honestly instead of at the converged number; and the ephemeris term follows
  //! the grade the tables served this cycle.
  F64 sigma_sun_sys_cycle_{0.0};
  F64 sigma_sun_total_cycle_{0.0};

  //! The Earth-albedo correction's per-unit constants (§8.1). Inert until
  //! refreshAlbedoConfig() succeeds; while inert every cycle is weighted at the
  //! uncorrected sigma, which is how the vehicle flew before it existed.
  //! The part-level terms (peak error, field of view). The per-unit boresight is
  //! filled in per cycle from @ref sun_boresights_ before the correction runs.
  polaris::gnc::AlbedoCorrectionConfig albedo_config_{};
  bool albedo_configured_{false};

  //! Per-unit albedo boresights in body axes, flattened three at a time in
  //! `sunSensorIn` port order (the `SunAlbedoBoresightsBody` parameter). A
  //! zero-vector slot means "no correction for this unit" — see
  //! @ref sunBoresightFor.
  F64 sun_boresights_[NUM_SUNSENSORIN_INPUT_PORTS * 3]{};

  //! Star-tracker fusion tuning (§8.2), the fourth independent validity gate.
  //! Inert until refreshStConfig() succeeds; while inert no tracker is fused and
  //! the ladder is capped at SS+MAG, which is how the vehicle flew before.
  bool st_configured_{false};
  FwIndexType st_king_unit_{0};
  F64 sigma_st_xy_rad_{0.0};
  F64 sigma_st_z_rad_{0.0};

  //! Per-unit star-tracker boresights in body axes, flattened three at a time in
  //! `starTrackerIn` port order (the `StBoresightsBody` parameter). A zero-vector
  //! slot means "not installed" — see @ref stBoresightFor.
  F64 st_boresights_[NUM_STARTRACKERIN_INPUT_PORTS * 3]{};

  //! Residual-monitor thresholds [rad], indexed as `ResidualMonitor`, and the
  //! persistence every monitor shares.
  F64 monitor_threshold_rad_[kMonitorCount]{};
  U32 monitor_alert_cycles_{0};

  //! Consecutive cycles each monitor has been over threshold, and whether it is
  //! currently in the alerted state (which is what makes the alert an edge and the
  //! recovery reportable).
  U32 monitor_streak_[kMonitorCount]{};
  bool monitor_alerted_[kMonitorCount]{};

  //! Which measurement source the fine solution is being updated from — the §8.2
  //! ladder's current rung. Changes are events, not demotions: the filter keeps
  //! its state across them.
  FineSource::T fine_source_{FineSource::NONE};

  //! **Per-unit** star-tracker NIS accounting. A cycle-global streak would let one
  //! persistently-disbelieved tracker demote the whole fine mode, which drops the
  //! filter and re-promotes off the same bad unit — a flap at the streak period.
  //! The unit is what gets isolated; the mode is not. `st_excluded_` latches a
  //! unit out of `collectStarTrackers`; `st_accepted_streak_` is its probation
  //! toward re-admission (@ref readmitStarTrackers).
  U32 st_nis_streak_[NUM_STARTRACKERIN_INPUT_PORTS]{};
  U32 st_accepted_streak_[NUM_STARTRACKERIN_INPUT_PORTS]{};
  bool st_excluded_[NUM_STARTRACKERIN_INPUT_PORTS]{};

  //! Fine-mode tuning cached from the parameter set alongside the MekfConfig.
  F64 seed_min_observability_{0.0};
  F64 bias_sigma_init_{0.0};
  U32 refusal_streak_limit_{0};
  U32 nis_streak_limit_{0};

  U32 triad_accepted_{0};
  U32 triad_rejected_{0};
  U32 cycles_refused_{0};
  U32 star_tracker_count_{0};
  U32 fine_demotions_{0};

  //! NIS-gate rejections accumulated over filters this component has already
  //! given up on. `Mekf::reset` clears the filter's own count by contract (a
  //! commanded reset means "start over"), which would erase the FDIR signal at
  //! the exact moment a NIS_STREAK or FILTER_FAULT demotion created it. Only
  //! RESET_ESTIMATOR clears this; telemetry reports it plus the live count.
  U32 fine_rejected_total_{0};

  //! Consecutive cycles the filter refused a call / the NIS gate rejected a
  //! measurement. Cleared by a clean cycle and by a demotion.
  U32 refusal_streak_{0};
  U32 nis_streak_{0};

  I64 last_epoch_tai_ns_{0};
  bool have_epoch_{false};

  //! Set at construction and by parameterUpdated(): the next cycle re-reads the
  //! whole parameter set before estimating.
  bool params_dirty_{true};

  //! Edge gates: each alert fires on the transition into its state, not once per
  //! cycle in it (the Push 38 TableDegraded pattern).
  bool attitude_valid_{false};
  bool config_invalid_flagged_{false};
  bool fine_config_invalid_flagged_{false};
  bool albedo_config_invalid_flagged_{false};
  bool st_config_invalid_flagged_{false};
  bool fine_init_failed_flagged_{false};
  bool imu_ambiguous_flagged_{false};
  bool mag_ambiguous_flagged_{false};

  //! The magnetometer vote's continuous-ambiguity run, mirroring the IMU one and
  //! sharing its horizon (`ImuAmbiguityEscalateCycles`). Separate counters because
  //! the two conditions can be present at once and cost different things.
  U32 mag_ambiguous_cycles_{0};
  I64 mag_ambiguous_start_ns_{0};

  //! Consecutive cycles the IMU vote has been unattributably ambiguous, the TAI
  //! epoch that run started at, and the horizon/cadence the escalation fires on
  //! (§9.2). The edge-gated alert alone would leave a *permanent* disagreement
  //! reported once and then silent for the rest of the flight.
  U32 imu_ambiguous_cycles_{0};
  I64 imu_ambiguous_start_ns_{0};
  U32 imu_ambiguity_escalate_cycles_{0};
  bool position_unavailable_flagged_{false};
  bool igrf_stale_flagged_{false};
  TableGrade::T last_ephem_grade_{TableGrade::PRECISE};
  TableGrade::T last_eop_grade_{TableGrade::PRECISE};
  bool have_ephem_grade_{false};
  bool have_eop_grade_{false};

  //! Solution age [s] of the last cycle, so the AttitudeLost edge can report it
  //! from paths that have no estimator output in hand (config failure).
  double last_age_s_{0.0};

  //! A collection window is open, and how many accepted samples close it. The
  //! two together are the whole of the calibration state machine: COLLECTING
  //! while open, APPLIED while `mag_cal_.valid`, IDLE otherwise.
  bool mag_cal_collecting_{false};
  U32 mag_cal_target_samples_{0};

  //! Cycles the open window has been given, and the deadline it is closed at.
  //! A window is defined in *accepted* samples, so a magnetometer or GNSS
  //! outage stalls it rather than ending it — without a deadline a loss of
  //! signal leaves the vehicle telemetering COLLECTING forever and the ground
  //! waiting on a completion that cannot arrive. The deadline is
  //! kMaxCalStallFactor x the target, i.e. the window may lose 90% of its
  //! cycles and still finish; past that the fit is attempted anyway, and the
  //! usual gates refuse it with SAMPLES if too little was collected.
  U32 mag_cal_cycles_{0};
  U32 mag_cal_deadline_cycles_{0};

  //! The inter-tracker alignment window: whether one is open, which unit it is
  //! calibrating against the king, and the same accepted-samples target plus
  //! stall deadline the magnetometer window carries and for the same reason — a
  //! window counted in *simultaneous pairs* is stalled rather than ended by a
  //! tracker outage.
  bool st_align_collecting_{false};
  FwIndexType st_align_unit_{0};
  U32 st_align_target_samples_{0};
  U32 st_align_cycles_{0};
  U32 st_align_deadline_cycles_{0};

  //! Fine mode is engaged: the MEKF is seeded and its solution is what
  //! estimateOut carries. False means the coarse chain is the published product.
  bool fine_active_{false};
};

}  // namespace flight

#endif
