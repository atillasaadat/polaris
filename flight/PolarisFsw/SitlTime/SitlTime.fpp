module flight {

  @ SITL sim-time provider: the FSW time source that serves sim time in a SITL
  @ run and wall-clock time otherwise (design doc §2.4, §3.2).
  @
  @ This is the deployment's single `time get port` source (replacing the stock
  @ Svc.ChronoTime). With SITL disabled it behaves exactly like ChronoTime:
  @ every timeGetPort call returns the workstation wall clock. With SITL enabled
  @ (setSitlActive() called at topology setup when --sitl-port is given), the
  @ SitlBridge pushes each macro-step's sim epoch on timeSetIn, and timeGetPort
  @ returns that epoch as TAI — so EVR/telemetry timestamps in a SITL run are
  @ pure functions of sim time, not wall time (§2.4 bit-reproducibility).
  passive component SitlTime {

    @ Time port: returns sim time (SITL active) or wall-clock time (otherwise).
    @ Declared directly (rather than `import Svc.Time`) so the topology autocoder
    @ needs only the Fw.Time port, not the Svc.Time interface symbol.
    sync input port timeGetPort: Fw.Time

    @ SitlBridge publishes the current macro-step sim epoch (TAI ns) here, once
    @ per STEP, before the rate group fires.
    sync input port timeSetIn: SitlTimeSet

  }

}
