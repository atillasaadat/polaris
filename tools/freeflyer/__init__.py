"""FreeFlyer integration: discovery, engine lifecycle, plan generation, V&V.

FreeFlyer (a.i. solutions) is the project's third, fully independent
astrodynamics implementation — alongside the Polaris C++ stack and the GMAT
golden fixtures — used for cross-validation (design doc §22) and for
interactive visualization of closed-loop SITL runs.

Nothing here is flight software. The package talks to a locally installed,
licensed FreeFlyer through the vendor's Runtime API Python client, which is
imported *from the installation itself* (:mod:`tools.freeflyer.locate`), so the
client version always matches the engine and no vendor code is vendored into
this repository.
"""
