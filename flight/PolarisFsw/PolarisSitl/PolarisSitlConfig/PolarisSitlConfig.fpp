module PolarisSitlConfig {
    # Base ID for the SITL subtopology; every instance is a fixed offset from it.
    #
    # Held at the original pre-subtopology flat-layout value (0x10015000) so the
    # command/telemetry/event dictionary is byte-identical across the Push 36
    # packaging change (design doc §2.2). 0x10015000 is exactly where the first
    # SITL instance (sitlBridge) sat when declared flat in Top/instances.fpp,
    # and the +0x1000-per-instance offsets reproduce the old flat IDs verbatim
    # rather than renumbering into a fresh subtopology block; the trade is a
    # documented convention deviation in exchange for a zero-diff dictionary
    # and green two-process gates.
    constant BASE_ID = 0x10015000
}
