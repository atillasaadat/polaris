"""Polaris config compiler (design doc §19.3). See compiler.py for the pipeline."""

from .compiler import (
    PRMDB_FILENAME,
    ConfigError,
    compile_config,
    emit_analysis_inputs,
    emit_fprime_params,
    emit_prmdb,
    emit_sim_setup,
    load_config,
    load_hardware_library,
    resolve,
)
from .orbit import MU_EARTH, keplerian_to_cartesian
from .prmdb import (
    ParamSpec,
    PrmDbError,
    build_param_file,
    encode_records,
    load_dictionary,
)
from .schema import Config, HardwareModel

__all__ = [
    "MU_EARTH",
    "PRMDB_FILENAME",
    "Config",
    "ConfigError",
    "HardwareModel",
    "ParamSpec",
    "PrmDbError",
    "build_param_file",
    "encode_records",
    "keplerian_to_cartesian",
    "compile_config",
    "emit_analysis_inputs",
    "emit_fprime_params",
    "emit_prmdb",
    "emit_sim_setup",
    "load_config",
    "load_dictionary",
    "load_hardware_library",
    "resolve",
]
