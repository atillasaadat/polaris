// ======================================================================
// \title  PolarisFswTopologyDefs.hpp
// \brief required header file containing the required definitions for the topology autocoder
//
// ======================================================================
#ifndef POLARISFSW_POLARISFSWTOPOLOGYDEFS_HPP
#define POLARISFSW_POLARISFSWTOPOLOGYDEFS_HPP

// Subtopology PingEntries includes
#include "Svc/Subtopologies/CdhCore/PingEntries.hpp"
#include "Svc/Subtopologies/ComCcsds/PingEntries.hpp"
#include "Svc/Subtopologies/DataProducts/PingEntries.hpp"
#include "Svc/Subtopologies/FileHandling/PingEntries.hpp"

// SubtopologyTopologyDefs includes
#include "Svc/Subtopologies/CdhCore/SubtopologyTopologyDefs.hpp"
#include "Svc/Subtopologies/ComCcsds/SubtopologyTopologyDefs.hpp"
#include "Svc/Subtopologies/DataProducts/SubtopologyTopologyDefs.hpp"
#include "Svc/Subtopologies/FileHandling/SubtopologyTopologyDefs.hpp"

// ComCcsds Enum Includes
#include "Svc/Subtopologies/ComCcsds/Ports_ComBufferQueueEnumAc.hpp"
#include "Svc/Subtopologies/ComCcsds/Ports_ComPacketQueueEnumAc.hpp"

// Include autocoded FPP constants
#include "flight/PolarisFsw/Top/FppConstantsAc.hpp"

/**
 * \brief required ping constants
 *
 * The topology autocoder requires a WARN and FATAL constant definition for each component that
 * supports the health-ping interface. These are expressed as enum constants placed in a namespace
 * named for the component instance. These are all placed in the PingEntries namespace.
 *
 * Each constant specifies how many missed pings are allowed before a WARNING_HI/FATAL event is
 * triggered. In the following example, the health component will emit a WARNING_HI event if the
 * component instance cmdDisp does not respond for 3 pings and will FATAL if responses are not
 * received after a total of 5 pings.
 *
 * ```c++
 * namespace PingEntries {
 * namespace cmdDisp {
 *     enum { WARN = 3, FATAL = 5 };
 * }
 * }
 * ```
 */
namespace PingEntries {
namespace flight_rateGroup1 {
enum { WARN = 3, FATAL = 5 };
}

namespace flight_rateGroup2 {
enum { WARN = 3, FATAL = 5 };
}

namespace flight_rateGroup3 {
enum { WARN = 3, FATAL = 5 };
}

namespace flight_cmdSeq {
enum { WARN = 3, FATAL = 5 };
}
}  // namespace PingEntries

// Definitions are placed within the same namespace as the FPP module that contains the topology.
namespace flight {

/**
 * \brief required type definition to carry state
 *
 * The topology autocoder requires an object that carries state with the name
 * `flight::TopologyState`. Only the type definition is required by the autocoder and the contents
 * of this object are otherwise opaque to the autocoder. The contents are entirely up to the
 * definition of the project. This deployment uses subtopologies.
 */
struct TopologyState {
  const char* hostname;  //!< Hostname for GDS TCP communication
  U16 port;              //!< Port for GDS TCP communication
  //! SITL/bench only: control mode to latch at startup (0 = leave IDLE), and the
  //! inertial-hold target that goes with it. On a flight vehicle both come from
  //! the ground or, from Phase 7, from the mode manager (design doc §8.5, §10).
  U32 ctrlMode;
  F64 ctrlTargetQ[4];
  //! SITL/bench only: an align/constrain pointing command to issue at startup
  //! (§8.4). `guidanceSet` false leaves the guidance uncommanded, which is the
  //! flight default — on a real vehicle this arrives by uplink. The fields are
  //! the SET_GUIDANCE arguments in order; see PointingGuidance.fpp.
  bool guidanceSet;
  U32 alignVecKind;
  U32 alignVecIndex;
  bool alignVecNegate;
  U32 alignTgtKind;
  U32 alignTgtIndex;
  bool alignTgtNegate;
  F64 alignTgtParam0;
  F64 alignTgtParam1;
  U32 conVecKind;
  U32 conVecIndex;
  bool conVecNegate;
  U32 conTgtKind;
  U32 conTgtIndex;
  bool conTgtNegate;
  F64 conTgtParam0;
  F64 conTgtParam1;
  //! SITL/bench only: a target-catalogue slot to fill at startup, so a
  //! `SAT_STATE_n` or `SAT_TLE_n` row can be flown with no ground link. Both are
  //! off by default (`< 0` / null), which is the flight default: a catalogue
  //! slot is uplinked, never born full.
  I32 satStateSlot;  //!< < 0 = do not load a state vector
  I64 satStateEpochTaiNs;
  F64 satStatePosM[3];
  F64 satStateVelMps[3];
  F64 satStateSigmaM;
  I32 satTleSlot;  //!< < 0 = do not load a TLE
  const char* satTleLine1;
  const char* satTleLine2;
  bool satTleVerifyChecksum;
  U16 sitlPort;                       //!< SITL lockstep port (0 = SITL disabled, §2.2)
  const char* onboardEopPath;         //!< Onboard IERS EOP table file (§11.3, §22)
  const char* onboardEphemPath;       //!< Onboard Chebyshev ephemeris fixture (§11.3, §22)
  const char* onboardIgrfPath;        //!< Onboard IAGA IGRF-14 coefficients (§6.2, §8.1)
  double igrfEpochYear;               //!< Mission epoch [decimal yr] the IGRF snapshot is taken
                                      //!< at; <= 0 = derive from the system clock at startup
  U32 magCalSamples;                  //!< SITL/bench only: sample count to command MAG_CAL_START
                                      //!< with at startup (0 = do not command a calibration).
                                      //!< The SITL demonstration of the §8.1 commanded flow needs
                                      //!< a command to arrive with no ground link attached; this
                                      //!< is that hook, and it dispatches the real opcode through
                                      //!< the component's own command port.
  U8 stAlignUnit;                     //!< SITL/bench only: starTrackerIn index the startup
                                      //!< ST_ALIGN_CAL_START names (ignored when the count is 0)
  U32 stAlignSamples;                 //!< SITL/bench only: simultaneous-pair count to command
                                      //!< ST_ALIGN_CAL_START with at startup (0 = do not command
                                      //!< an alignment calibration). The §8.2 twin of
                                      //!< magCalSamples, and there for the same reason.
  I32 ffModel = -1;                   //!< SITL/bench only: force the §8.5 tier-1 feedforward on
                                      //!< (1) or off (0); negative leaves the ParameterDb value.
                                      //!< Exists for one experiment — flying the same vehicle
                                      //!< with and without feedforward is the only way to measure
                                      //!< what it buys (design doc §8.5). Defaulted so an entry
                                      //!< point that never sets it flies the ParameterDb value
                                      //!< rather than an indeterminate override.
  I32 ffObserver = -1;                //!< SITL/bench only: the tier-2 twin of ffModel, same
                                      //!< default for the same reason. The observer still runs
                                      //!< when this is 0; it is also the §9 anomaly monitor.
  I32 wheelBias = -1;                 //!< SITL/bench only: force the §8.5 wheel-speed bias on
                                      //!< (1) or off (0); negative leaves the ParameterDb
                                      //!< pattern. The zero-crossing A/B row's switch.
  U32 burnStartCycle = 0;             //!< SITL/bench only: GNC cycle on which to command
                                      //!< BURN_START (0 = never); with the two below, the
                                      //!< §17 burn rows' way to fire a burn with no ground link.
  F64 burnDurationS = 0.0;            //!< SITL/bench only: BURN_START duration [s]
  F64 burnThrottle = 0.0;             //!< SITL/bench only: BURN_START throttle (0, 1]
  I32 odAccelInput = -1;              //!< SITL/bench only: 0 = the orbit filter ignores the
                                      //!< burn executor's acceleration (the "blind" half of
                                      //!< the burn-in-outage A/B); negative/1 = flight behaviour.
  U32 odResetCycle = 0;               //!< SITL/bench only: GNC cycle on which to command the
                                      //!< orbit filter's OD_RESET (0 = never). The §8.3
                                      //!< reset-and-reseed row needs a command mid-run with no
                                      //!< ground link; it dispatches the real opcode through the
                                      //!< component's own command port, like magCalSamples.
  const char* prmDbPath;              //!< ParameterDb file emitted by the config compiler
                                      //!< (§19.3); nullptr = the FileHandling default "PrmDb.dat"
  CdhCore::SubtopologyState cdhCore;  //!< Subtopology state for CdhCore
  ComCcsds::SubtopologyState comCcsds;          //!< Subtopology state for ComCcsds
  DataProducts::SubtopologyState dataProducts;  //!< Subtopology state for DataProducts
  FileHandling::SubtopologyState fileHandling;  //!< Subtopology state for FileHandling
};

namespace PingEntries = ::PingEntries;
}  // namespace flight

#endif
