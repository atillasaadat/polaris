// ======================================================================
// \title  PolarisFswTopology.cpp
// \brief cpp file containing the topology instantiation code
//
// ======================================================================
// Provides access to autocoded functions
#include <flight/PolarisFsw/Top/PolarisFswTopologyAc.hpp>
// Note: Uncomment when using Svc:TlmPacketizer
// #include <flight/PolarisFsw/Top/PolarisFswPacketsAc.hpp>

// Necessary project-specified types
#include <Fw/Types/MallocAllocator.hpp>

// SITL comm stack configuration types (design doc §2.2, §2.4)
#include <cstring>
#include <Svc/FrameAccumulator/FrameDetector/FprimeFrameDetector.hpp>

#include "sitl/wire.hpp"

// Public functions for use in main program are namespaced with deployment module flight
// This is also the namespace where the topology components are instantiated by FPP.
namespace flight {

// Instantiate a malloc allocator for cmdSeq buffer allocation
Fw::MallocAllocator mallocator;

// SITL lockstep transport (§2.2, §2.4). The frame detector must persist for the
// life of frameAccumulatorSitl; a shared malloc allocator backs the SITL buffer
// pool (allocated once at init — no steady-state heap).
Svc::FrameDetectors::FprimeFrameDetector sitlFrameDetector;
Fw::MallocAllocator sitlAllocator;

// SITL buffer pool: one bin sized to the largest single allocation — a whole
// deframed STEP_REQ frame (kMaxStepReqBytes + F´ frame header/trailer). The
// TcpClient recv buffers and the framer's reply frames are smaller and fit too.
enum SitlConstants {
  SITL_BUFFER_SIZE = polaris::sitl::kMaxStepReqBytes + 64,
  SITL_BUFFER_COUNT = 10,
  SITL_ACCUMULATOR_SIZE = SITL_BUFFER_SIZE * 2,
  SITL_COMM_PRIORITY = 33,
};

// The reference topology divides the incoming clock signal (1Hz) into sub-signals: 1Hz, 1/2Hz, and
// 1/4Hz with 0 offset
Svc::RateGroupDriver::DividerSet rateGroupDivisorsSet{{{1, 0}, {2, 0}, {4, 0}}};

// Rate groups may supply a context token to each of the attached children whose purpose is set by
// the project. The reference topology sets each token to zero as these contexts are unused in this
// project.
U32 rateGroup1Context[Svc::ActiveRateGroup::CONNECTION_COUNT_MAX] = {};
U32 rateGroup2Context[Svc::ActiveRateGroup::CONNECTION_COUNT_MAX] = {};
U32 rateGroup3Context[Svc::ActiveRateGroup::CONNECTION_COUNT_MAX] = {};

enum TopologyConstants {
  COMM_PRIORITY = 34,
};

/**
 * \brief configure/setup components in project-specific way
 *
 * This is a *helper* function which configures/sets up each component requiring project specific
 * input. This includes allocating resources, passing-in arguments, etc. This function may be
 * inlined into the topology setup function if desired, but is extracted here for clarity.
 */
void configureTopology() {
  // Rate group driver needs a divisor list
  rateGroupDriver.configure(rateGroupDivisorsSet);

  // Rate groups require context arrays.
  rateGroup1.configure(rateGroup1Context, FW_NUM_ARRAY_ELEMENTS(rateGroup1Context));
  rateGroup2.configure(rateGroup2Context, FW_NUM_ARRAY_ELEMENTS(rateGroup2Context));
  rateGroup3.configure(rateGroup3Context, FW_NUM_ARRAY_ELEMENTS(rateGroup3Context));

  // Command sequencer needs to allocate memory to hold contents of command sequences
  cmdSeq.allocateBuffer(0, mallocator, 5 * 1024);

  // SITL comm stack: frame detector + buffer pool. Configured unconditionally so
  // the instances are valid; they stay inert until the TcpClient is started (only
  // when --sitl-port is given), so a SITL-off run behaves exactly as before.
  frameAccumulatorSitl.configure(sitlFrameDetector, 1, sitlAllocator, SITL_ACCUMULATOR_SIZE);

  Svc::BufferManager::BufferBins sitlBins;
  memset(&sitlBins, 0, sizeof(sitlBins));
  sitlBins.bins[0].bufferSize = SITL_BUFFER_SIZE;
  sitlBins.bins[0].numBuffers = SITL_BUFFER_COUNT;
  commsBufferManagerSitl.setup(0, 0, sitlAllocator, sitlBins);
}

void setupTopology(const TopologyState& state) {
  // Autocoded initialization. Function provided by autocoder.
  initComponents(state);
  // Autocoded id setup. Function provided by autocoder.
  setBaseIds();
  // Autocoded connection wiring. Function provided by autocoder.
  connectComponents();
  // Autocoded command registration. Function provided by autocoder.
  regCommands();
  // Autocoded configuration. Function provided by autocoder.
  configComponents(state);
  if (state.hostname != nullptr && state.port != 0) {
    comDriver.configure(state.hostname, state.port);
  }
  // SITL link connects to the truth sim (which listens) on loopback (§2.2). The
  // FrameAccumulator reassembles frames across recv buffers, so the default recv
  // buffer size is sufficient.
  if (state.sitlPort != 0) {
    comDriverSitl.configure("127.0.0.1", state.sitlPort);
  }
  // Project-specific component configuration. Function provided above. May be inlined, if desired.
  configureTopology();
  // Autocoded parameter loading. Function provided by autocoder.
  loadParameters();
  // Autocoded task kick-off (active components). Function provided by autocoder.
  startTasks(state);
  // Initialize socket communication if and only if there is a valid specification
  if (state.hostname != nullptr && state.port != 0) {
    Os::TaskString name("ReceiveTask");
    // Uplink is configured for receive so a socket task is started
    comDriver.start(name, COMM_PRIORITY, Default::STACK_SIZE);
  }
  // Start the SITL receive task only when a SITL port was given; otherwise the
  // stack stays inert (design doc §2.2).
  if (state.sitlPort != 0) {
    Os::TaskString sitlName("SitlRecvTask");
    comDriverSitl.start(sitlName, SITL_COMM_PRIORITY, Default::STACK_SIZE);
  }
}

void startRateGroups(const Fw::TimeInterval& interval) {
  // The timer component drives the fundamental tick rate of the system.
  // Svc::RateGroupDriver will divide this down to the slower rate groups.
  // This call will block until the stopRateGroups() call is made.
  // For this Linux demo, that call is made from a signal handler.
  timer.startTimer(interval);
}

void stopRateGroups() {
  timer.quit();
}

void teardownTopology(const TopologyState& state) {
  // Autocoded (active component) task clean-up. Functions provided by topology autocoder.
  stopTasks(state);
  freeThreads(state);

  // Other task clean-up.
  comDriver.stop();
  (void)comDriver.join();
  comDriverSitl.stop();
  (void)comDriverSitl.join();

  // Resource deallocation
  cmdSeq.deallocateBuffer(mallocator);
  frameAccumulatorSitl.cleanup();
  commsBufferManagerSitl.cleanup();

  tearDownComponents(state);
  deinitComponents(state);
}
};  // namespace flight
