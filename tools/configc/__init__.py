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
from .schema import Config, HardwareModel

__all__ = [
    "Config",
    "ConfigError",
    "HardwareModel",
    "compile_config",
    "emit_analysis_inputs",
    "emit_fprime_params",
    "emit_sim_setup",
    "load_config",
    "load_hardware_library",
    "resolve",
]
