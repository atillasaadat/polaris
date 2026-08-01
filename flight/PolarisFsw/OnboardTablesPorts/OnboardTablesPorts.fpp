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

  @ Source quality of a table query answer (design doc §8.1, §11.3 coarse-
  @ fallback note). Callers gate on this so table loss degrades to coarse
  @ operation instead of no answer: PRECISE = uploaded table; COARSE = table-
  @ independent fallback (analytic Vallado Sun/Moon, or zero-EOP with UT1 approx
  @ UTC and zero polar motion) — the piece that makes the Safe-mode coarse
  @ sun-pointing floor table-independent; UNAVAILABLE = no answer (reserved).
  enum TableGrade : U8 {
    UNAVAILABLE = 0
    COARSE = 1
    PRECISE = 2
  }

  @ Table domain a grade transition applies to, for the degrade/recover alerts.
  enum TableDomain : U8 {
    EPHEMERIS = 0
    EOP = 1
  }

  @ IERS Earth-orientation parameters interpolated to a query epoch, in the
  @ units the ECI<->ECEF reduction consumes (UT1-TAI is continuous across leap
  @ seconds; polar motion in arcseconds).
  struct EopSample {
    ut1MinusTai: F64 @< UT1 - TAI [s]
    xpArcsec: F64 @< polar motion x [arcsec]
    ypArcsec: F64 @< polar motion y [arcsec]
    grade: TableGrade @< source quality (PRECISE table vs COARSE zero-EOP fallback)
  }

  @ A geocentric ECI (ICRF/J2000) position [m].
  struct PosEciMeters {
    x: F64
    y: F64
    z: F64
    grade: TableGrade @< source quality (PRECISE Chebyshev vs COARSE analytic fallback)
  }

  @ EOP at a TAI epoch. Always answers: sample is written and true returned; the
  @ grade field says whether it came from the uploaded table (PRECISE) or the
  @ zero-EOP coarse fallback (COARSE). Returns false only in the reserved
  @ UNAVAILABLE case (does not occur for EOP).
  port GetEopAt(
                 taiNs: I64 @< query epoch, TAI nanoseconds since 1970-01-01
                 ref sample: EopSample @< interpolated EOP + source grade
               ) -> bool

  @ Geocentric ECI position [m] of a body at a TAI epoch. Always answers:
  @ position is written and true returned; the grade field says whether it came
  @ from the uploaded Chebyshev fit (PRECISE) or the analytic Vallado fallback
  @ (COARSE). Returns false only in the reserved UNAVAILABLE case.
  port GetBodyPosition(
                        body: OnboardBody
                        taiNs: I64 @< query epoch, TAI nanoseconds
                        ref posEciM: PosEciMeters @< ECI [m] + source grade
                      ) -> bool

  @ TAI - UTC (delta-AT) [s] at a TAI epoch. Always answers (never false): the
  @ in-code IERS leap record is populated at construction, independent of any
  @ table load — the time chain works even if every upload fails (Safe-mode
  @ floor guarantee).
  port GetTaiUtcOffset(
                        taiNs: I64 @< query epoch, TAI nanoseconds
                        ref deltaAtSec: I32 @< TAI - UTC [s], valid only on true
                      ) -> bool

}
