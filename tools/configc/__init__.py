"""Polaris config compiler (design doc §19.3). See compiler.py for the pipeline."""

from .compiler import (
    ConfigError,
    compile_config,
    emit_analysis_inputs,
    emit_fprime_params,
    emit_sim_setup,
    load_config,
    load_hardware_library,
    resolve,
)
from .orbit import MU_EARTH, keplerian_to_cartesian
from .schema import Config, HardwareModel

__all__ = [
    "MU_EARTH",
    "Config",
    "ConfigError",
    "HardwareModel",
    "keplerian_to_cartesian",
    "compile_config",
    "emit_analysis_inputs",
    "emit_fprime_params",
    "emit_sim_setup",
    "load_config",
    "load_hardware_library",
    "resolve",
]
