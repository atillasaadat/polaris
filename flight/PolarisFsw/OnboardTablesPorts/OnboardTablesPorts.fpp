module flight {

  # ----------------------------------------------------------------------
  # Shared onboard-table query port/type definitions (design doc §11.3, §22).
  #
  # The typed seam future Phase-4 GNC components (TRIAD/QUEST/MEKF, ECI<->ECEF
  # on ingest, sun-vector reference) use to read the onboard tables the
  # OnboardTables component serves. Kept in one interface-only module so the
  # component and every consumer share exactly one definition and cannot drift.
  # Query epochs are TAI nanoseconds (the FSW master clock, §3.2); positions are
  # geocentric ECI (ICRF/J2000) metres.
  # ----------------------------------------------------------------------

  @ Solar-system bodies the onboard Chebyshev ephemeris carries.
  enum OnboardBody : U8 {
    SUN = 0
    MOON = 1
  }

  @ IERS Earth-orientation parameters interpolated to a query epoch, in the
  @ units the ECI<->ECEF reduction consumes (UT1-TAI is continuous across leap
  @ seconds; polar motion in arcseconds).
  struct EopSample {
    ut1MinusTai: F64 @< UT1 - TAI [s]
    xpArcsec: F64 @< polar motion x [arcsec]
    ypArcsec: F64 @< polar motion y [arcsec]
  }

  @ A geocentric ECI (ICRF/J2000) position [m].
  struct PosEciMeters {
    x: F64
    y: F64
    z: F64
  }

  @ EOP at a TAI epoch. Returns false (sample untouched) if no tables are loaded
  @ or the epoch is outside the EOP coverage span (no extrapolation, §3.6).
  port GetEopAt(
                 taiNs: I64 @< query epoch, TAI nanoseconds since 1970-01-01
                 ref sample: EopSample @< interpolated EOP, valid only on true
               ) -> bool

  @ Geocentric ECI position [m] of a body at a TAI epoch. Returns false
  @ (position untouched) if no tables are loaded or the epoch is uncovered.
  port GetBodyPosition(
                        body: OnboardBody
                        taiNs: I64 @< query epoch, TAI nanoseconds
                        ref posEciM: PosEciMeters @< ECI [m], valid only on true
                      ) -> bool

  @ TAI - UTC (delta-AT) [s] at a TAI epoch. Returns false only if no tables are
  @ loaded; once loaded the leap table always answers.
  port GetTaiUtcOffset(
                        taiNs: I64 @< query epoch, TAI nanoseconds
                        ref deltaAtSec: I32 @< TAI - UTC [s], valid only on true
                      ) -> bool

}
