! Polaris <-> NRLMSIS 2.1 C interoperability shim.
!
! NRLMSIS 2.1 exposes Fortran 90 module procedures with assumed-shape and
! optional arguments, none of which are callable from C++. This file wraps the
! two entry points Polaris needs in `bind(C)` procedures with flat, explicitly
! sized arguments.
!
! This is Polaris's own source, NOT a modification of NRL's. That distinction is
! deliberate: the MSIS license (clause 4b) requires that modifications and
! derivative works of the model itself carry change notices and be delivered
! back to NRL Code 7630. A separate wrapper that only *calls* the published
! interface incurs none of that. Nothing under `${nrlmsis_SOURCE_DIR}` is ever
! edited — it is fetched read-only at configure time (cmake/Dependencies.cmake).
!
! This software incorporates the MSIS(R) empirical atmospheric model software
! designed and provided by NRL. Use is governed by the Open Source Academic
! Research License Agreement contained in the file nrlmsis2.1_license.txt.

!> Initialize the model and load the coefficient file.
!! @param path      Null-terminated path to msis21.parm.
!! @param path_len  Length of `path` excluding the terminator.
!! @param status    0 on success. Nonzero means the parameter file could not be
!!                  loaded, and no density call may follow.
subroutine polaris_msis_init(path, path_len, status) bind(C, name="polaris_msis_init")
  use, intrinsic :: iso_c_binding, only: c_char, c_int
  use msis_init, only: msisinit
  implicit none

  character(kind=c_char), intent(in) :: path(*)
  integer(c_int), value, intent(in)  :: path_len
  integer(c_int), intent(out)        :: status

  character(len=:), allocatable :: parmpath
  integer :: i

  ! MSIS takes the directory and filename separately and concatenates them, so
  ! the whole path is handed over as the "path" with an empty filename.
  allocate(character(len=path_len) :: parmpath)
  do i = 1, path_len
    parmpath(i:i) = path(i)
  end do

  status = 0
  ! A missing or malformed parameter file makes msisinit stop the process, which
  ! is unacceptable inside a simulation. The C++ side checks readability before
  ! calling here, so reaching msisinit with a bad file is already a bug.
  call msisinit(parmpath=parmpath, parmfile='')

  deallocate(parmpath)
end subroutine polaris_msis_init

!> Legacy `gtd8d` entry point, used only to validate this build of the model
!! against NRL's own shipped reference output (msis2.1_test_ref_dp.txt), which
!! was produced through this interface. Note its arguments are single precision
!! even in the -DDBLE build, matching upstream's declaration.
!!
!! @param d_out  The full 10-element legacy density array. Element 6 is the total
!!               mass density in **g/cm^3** (legacy units, not SI).
subroutine polaris_msis_gtd8d(iyd, sec, alt, glat, glong, stl, f107a, f107, ap, d_out) &
    bind(C, name="polaris_msis_gtd8d")
  use, intrinsic :: iso_c_binding, only: c_float, c_int
  implicit none

  integer(c_int), value, intent(in) :: iyd
  real(c_float), value, intent(in)  :: sec, alt, glat, glong, stl, f107a, f107
  real(c_float), intent(in)         :: ap(7)
  real(c_float), intent(out)        :: d_out(10)

  real(4) :: t(2)
  integer, parameter :: mass = 48   ! upstream's "all species" selector

  call gtd8d(iyd, sec, alt, glat, glong, stl, f107a, f107, ap, mass, d_out, t)
end subroutine polaris_msis_gtd8d

!> Total neutral mass density at one point.
!!
!! @param day       Day of year [1, 366], fractional part ignored by the model.
!! @param utsec     Seconds into the UT day.
!! @param alt_km    Geodetic altitude [km].
!! @param lat_deg   Geodetic latitude [deg].
!! @param lon_deg   Longitude [deg east].
!! @param f107a     81-day centered average F10.7 solar flux [sfu].
!! @param f107      Daily F10.7 for the previous day [sfu].
!! @param ap        7-element Ap array (see msis_calc.F90 for the layout).
!! @param density   Total mass density [kg/m^3]. Set to 0 if MSIS returns its
!!                  missing-value sentinel (9.999e-38) or a non-finite result,
!!                  so callers see "no atmosphere" rather than a poisoned number.
subroutine polaris_msis_density(day, utsec, alt_km, lat_deg, lon_deg, &
                                f107a, f107, ap, density) bind(C, name="polaris_msis_density")
  use, intrinsic :: iso_c_binding, only: c_double
  use msis_calc, only: msiscalc
  implicit none

  real(c_double), value, intent(in) :: day, utsec, alt_km, lat_deg, lon_deg
  real(c_double), value, intent(in) :: f107a, f107
  real(c_double), intent(in)        :: ap(7)
  real(c_double), intent(out)       :: density

  real(kind=8) :: tn, dn(10)

  call msiscalc(day, utsec, alt_km, lat_deg, lon_deg, f107a, f107, ap, tn, dn)

  density = dn(1)
  ! Reject anything that is not a real density: 9.999e-38 is MSIS's documented
  ! missing-value marker (msis_calc.F90 header), and `density /= density` is true
  ! only for NaN.
  if (.not. (density > 1.0d-37) .or. density /= density) density = 0.0d0
end subroutine polaris_msis_density
