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
 * **Why 128, and how much of it is left.** The reference vehicle reached 26
 * parameters at Push 46 against the framework default of 25, which is what forced
 * this file; Push 51 took it to 37 and Push 52's star-tracker fusion, mode-ladder
 * monitors, magnetometer voting and alignment calibration took it to 55 of 64 —
 * nine spare. Push 54's attitude control adds 30 (B-dot, the pointing PID, the
 * wheel allocation and the MTQ/MAG interlock), which is what the Push 52 note
 * predicted would happen, so the limit moved to **128** rather than being shaved
 * against: the vehicle now sits at **85 of 128**.
 *
 * The cost is a `Fw::ArrayMap` of 128 parameter buffers, statically allocated,
 * comfortably inside the deployment's memory budget. Phases 6-10 still owe orbit
 * determination, FDIR, CFDP and sequencing tuning; at 43 spare that is headroom
 * rather than a countdown. Only this number needs changing:
 * `tools/configc/prmdb.py` parses it out of this file, and refuses to emit a
 * longer one — so overflowing it is a build error rather than a partially-loaded
 * database in flight.
 */

#ifndef PRMDB_PRMDBLIMPLCFG_HPP_
#define PRMDB_PRMDBLIMPLCFG_HPP_

// Anonymous namespace for configuration parameters
namespace {

enum {
  PRMDB_NUM_DB_ENTRIES = 128,   // !< Number of entries in the parameter database
  PRMDB_ENTRY_DELIMITER = 0xA5  // !< Byte value that should precede each parameter in file; sanity
                                // check against file integrity. Should match ground system.
};

}

#endif /* PRMDB_PRMDBLIMPLCFG_HPP_ */
