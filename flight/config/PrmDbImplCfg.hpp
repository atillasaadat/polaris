/*
 * PrmDbImplCfg.hpp — Polaris override of the F´ default (design doc §19.3).
 *
 * Overrides fprime/default/config/PrmDbImplCfg.hpp via CONFIGURATION_OVERRIDES;
 * everything except the entry count is the framework default, byte for byte.
 *
 * **Why this file exists.** `Svc::PrmDb` sizes its store at compile time, and a
 * parameter file longer than the store loads *partially* — the records past the
 * limit are silently dropped and the components that own them come up with
 * INVALID parameters. On a vehicle with no flight defaults (§19.3) that is a
 * vehicle with no attitude solution, reported as a missing parameter rather than
 * as a full database. `tools/configc/prmdb.py` therefore refuses to emit a file
 * longer than this limit, which is what turns the failure into a build error;
 * that module's MAX_ENTRIES must stay equal to PRMDB_NUM_DB_ENTRIES below.
 *
 * **Why 64.** The reference vehicle reached 26 parameters at Push 46 (twelve
 * coarse-attitude, seven fine-mode, seven magnetometer-calibration) against the
 * framework default of 25, and Phase 4-6 still owe control, orbit determination
 * and FDIR tuning. 64 is the next power of two with room for those without
 * revisiting this file each push; the cost is a `Fw::ArrayMap` of 64 parameter
 * buffers, statically allocated, which is comfortably inside the deployment's
 * memory budget.
 */

#ifndef PRMDB_PRMDBLIMPLCFG_HPP_
#define PRMDB_PRMDBLIMPLCFG_HPP_

// Anonymous namespace for configuration parameters
namespace {

enum {
  PRMDB_NUM_DB_ENTRIES = 64,    // !< Number of entries in the parameter database
  PRMDB_ENTRY_DELIMITER = 0xA5  // !< Byte value that should precede each parameter in file; sanity
                                // check against file integrity. Should match ground system.
};

}

#endif /* PRMDB_PRMDBLIMPLCFG_HPP_ */
