# flight/config — Polaris overrides of the F´ default configuration.
#
# Only the files this project actually changes live here; everything else comes
# from fprime/default/config unchanged. CONFIGURATION_OVERRIDES copies each file
# over the framework's own copy in the build cache, so there is no vendored fork
# of the F´ config directory to keep in step with the submodule.
#
# Override order is platform -> fprime config -> library -> project, so this
# directory must be registered *after* the framework config, which it is: the
# project CMakeLists adds flight/ after fprime_setup_included_code().

register_fprime_config(
    PolarisFprimeConfig
  CONFIGURATION_OVERRIDES
    "${CMAKE_CURRENT_LIST_DIR}/PrmDbImplCfg.hpp"
  # INTERFACE because the module contributes no compiled source of its own: it
  # only replaces a header inside the framework's config build.
  INTERFACE
)
