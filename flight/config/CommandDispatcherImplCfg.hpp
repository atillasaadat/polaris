/*
 * CommandDispatcherImplCfg.hpp — Polaris override of the F´ default.
 *
 * Overrides fprime/default/config/CommandDispatcherImplCfg.hpp via
 * CONFIGURATION_OVERRIDES; the sequencer table is the framework default, byte
 * for byte.
 *
 * **Why this file exists.** `Svc::CmdDispatcher` sizes its opcode table at
 * compile time, and registration past the limit is an `FW_ASSERT` in
 * `compCmdReg_handler` — i.e. the deployment aborts during topology setup, after
 * emitting 150 perfectly healthy OpCodeRegistered events. That is a loud failure
 * rather than a silent one, which is right, but it fires at the moment a
 * component is *added*, so it deserves the same one-line override the parameter
 * database got rather than a vendored fork of the F´ config directory.
 *
 * **Why 512.** The deployment reached the framework's 150 at Push 52, when the
 * attitude estimator's three ST_ALIGN_CAL_* commands landed on top of the F´
 * service components' own (`CmdDispatcher`, `EventManager`, `TlmChan`,
 * `PrmDb`, `FileDownlink`, `FileManager`, `FileUplink`, `CmdSequencer`,
 * `DpCatalog`, `Version`, and the two Polaris components); 256 lasted to
 * Push 56, whose 44th controller parameter was opcode number 257 — every
 * parameter costs *two* opcodes (PRM_SET and PRM_SAVE), so the table fills at
 * roughly twice the parameter count plus the explicit commands, and 100
 * parameters were already 200 opcodes. Phases 5-10 still owe control,
 * orbit-determination, FDIR, CFDP and time-tagged-sequencing commands *and*
 * their parameters, so the next power of two is the value that stops this file
 * needing a revisit each push. The cost is a `Fw::RedBlackTreeMap` node array of
 * 512 {opcode, port} pairs — a few kilobytes, statically allocated, comfortably
 * inside the deployment's memory budget.
 */

#ifndef CMDDISPATCHER_COMMANDDISPATCHERIMPLCFG_HPP_
#define CMDDISPATCHER_COMMANDDISPATCHERIMPLCFG_HPP_

// Define configuration values for dispatcher

enum {
  CMD_DISPATCHER_DISPATCH_TABLE_SIZE = 512,  // !< The size of the table holding opcodes to dispatch
  CMD_DISPATCHER_SEQUENCER_TABLE_SIZE =
      25,  // !< The size of the table holding commands in progress
};

#endif /* CMDDISPATCHER_COMMANDDISPATCHERIMPLCFG_HPP_ */
