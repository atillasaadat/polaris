module flight {

  @ Onboard time/EOP/ephemeris table provider (design doc §11.3, §22;
  @ REQ-CDH-002).
  @
  @ Loads the three onboard reference tables at topology setup and serves point
  @ queries against them to the Phase-4 GNC stack: the leap-second (delta-AT)
  @ table, the IERS EOP table, and the Chebyshev ephemeris of the Sun and Moon.
  @ Wraps the flight-safe lib holder (lib/onboard, which wraps the lib
  @ tables/evaluators) — no parsing or math lives here.
  @
  @ Tables are files on disk; persistence across restart is the filesystem. The
  @ operational upload path is F´ FileUplink (GDS writes a new table file) then
  @ the RELOAD_TABLES command, which restages and re-validates: the lib holder
  @ parses into an inactive double-buffer slot and swaps only on full success, so
  @ a failed upload/reload leaves the previous tables in service and a query
  @ never sees a half-loaded table.
  @
  @ Passive: the query ports are lock-free bounded reads of the active slot (the
  @ swap is a single atomic index flip), so no queue is needed. Not a SITL
  @ component — it ships to hardware and has no lib/sitl dependency.
  passive component OnboardTables {

    # ----------------------------------------------------------------------
    # Query ports (thin sync wrappers over the lib evaluators)
    # ----------------------------------------------------------------------

    @ EOP at a TAI epoch (UT1-TAI, polar motion), for the ECI<->ECEF reduction.
    sync input port getEopAt: GetEopAt

    @ Geocentric ECI position [m] of the Sun or Moon at a TAI epoch, for the
    @ sun-vector reference and third-body models.
    sync input port getBodyPosition: GetBodyPosition

    @ TAI - UTC (delta-AT) [s] at a TAI epoch, for UTC-facing output.
    sync input port getTaiUtcOffset: GetTaiUtcOffset

    # ----------------------------------------------------------------------
    # Scheduler: coverage-expiry check on a wall-clock rate group
    # ----------------------------------------------------------------------

    @ Rate-group entry: check the current time is still inside the EOP and
    @ ephemeris coverage and warn (throttled) if it has run past either.
    sync input port run: Svc.Sched

    # ----------------------------------------------------------------------
    # Commands
    # ----------------------------------------------------------------------

    @ Reload all tables from their configured paths (upload -> activate). Safe:
    @ restages into an inactive buffer and swaps only on full success.
    sync command RELOAD_TABLES

    # ----------------------------------------------------------------------
    # Telemetry (health: counts, coverage spans, load state)
    # ----------------------------------------------------------------------

    @ True once a full load has succeeded and queries can answer.
    telemetry TablesValid: bool

    @ Leap-second (delta-AT) table entries loaded.
    telemetry LeapEntries: U32

    @ IERS EOP daily records loaded (windowed to the ephemeris span).
    telemetry EopEntries: U32

    @ Sun Chebyshev segments loaded.
    telemetry SunSegments: U32

    @ Moon Chebyshev segments loaded.
    telemetry MoonSegments: U32

    @ EOP coverage start, TAI seconds since 1970-01-01.
    telemetry EopStartTai: I64

    @ EOP coverage end, TAI seconds since 1970-01-01.
    telemetry EopEndTai: I64

    @ Ephemeris coverage start, TAI seconds since 1970-01-01.
    telemetry EphStartTai: I64

    @ Ephemeris coverage end, TAI seconds since 1970-01-01.
    telemetry EphEndTai: I64

    @ Successful (re)loads since start.
    telemetry ReloadCount: U32

    # ----------------------------------------------------------------------
    # Events
    # ----------------------------------------------------------------------

    @ Leap-second table loaded.
    event LeapTableLoaded(entries: U32) \
      severity activity high \
      format "Onboard leap-second table loaded: {} entries"

    @ EOP table loaded, with coverage span (TAI seconds since epoch).
    event EopTableLoaded(entries: U32, startTai: I64, endTai: I64) \
      severity activity high \
      format "Onboard EOP table loaded: {} records, coverage [{}, {}] TAI s"

    @ Ephemeris tables loaded, with Sun/Moon segment counts and coverage span.
    event EphemerisTableLoaded(sunSegments: U32, moonSegments: U32, startTai: I64, endTai: I64) \
      severity activity high \
      format "Onboard ephemeris loaded: sun={} moon={} segments, coverage [{}, {}] TAI s"

    @ A table failed to load; the reason names which table and why. On a reload
    @ the previous tables stay in service (nothing was swapped in).
    event TableLoadFailed(reason: string size 80) \
      severity warning high \
      format "Onboard table load failed: {}"

    @ All tables reloaded successfully by operator command.
    event TablesReloaded(reloadCount: U32) \
      severity activity high \
      format "Onboard tables reloaded (reload #{})"

    @ The current time has run past the EOP and/or ephemeris coverage: the
    @ affected queries will start returning invalid. Throttled so it flags once
    @ rather than every rate-group cycle.
    event CoverageExpiring(eopExpired: bool, ephemExpired: bool) \
      severity warning high \
      format "Onboard table coverage expiring: eop={} ephem={}" \
      throttle 1

    # ----------------------------------------------------------------------
    # Standard AC ports
    # ----------------------------------------------------------------------

    @ Port for requesting the current time (TAI under SITL, via SitlTime).
    time get port timeCaller

    @ Command registration port
    command reg port cmdRegOut

    @ Command receive port
    command recv port cmdIn

    @ Command response port
    command resp port cmdResponseOut

    @ Event port
    event port logOut

    @ Text event port
    text event port logTextOut

    @ Telemetry port
    telemetry port tlmOut

  }

}
