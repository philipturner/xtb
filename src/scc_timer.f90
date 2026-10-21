! This file is part of xtb.
!
! xtb is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! xtb is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with xtb.  If not, see <https://www.gnu.org/licenses/>.

!> Wall-clock timers for the stages of one SCC cycle (profiling only).
!>
!> Disabled unless the environment variable XTB_SCC_TIMERS names a file. Then
!> every SCC cycle appends one line to that file:
!>   cycle <ndim> <iter> <potential> <hbuild> <eigensolve> <fermi> <dmat> <population> <mixing>
!> and every call of the SCC iterator appends
!>   setup <ndim> <factorize S>
!>   done <ndim> <cycles> <loop total>
!> All times in seconds; the loop total includes the setup and excludes the
!> time spent writing this file. Timing does not change any computed value.
module xtb_scc_timer
   use xtb_mctc_accuracy, only : wp
   use, intrinsic :: iso_fortran_env, only : int64
   implicit none
   private

   public :: scc_timer_start, scc_timer_lap, scc_timer_setup, scc_timer_cycle, &
      & scc_timer_done

   !> Stages of one SCC cycle
   integer, parameter, public :: stage_potential = 1, stage_hbuild = 2, &
      & stage_eigensolve = 3, stage_fermi = 4, stage_dmat = 5, &
      & stage_population = 6, stage_mixing = 7
   integer, parameter :: nstage = 7

   logical, save :: initialized = .false.
   logical, save :: enabled = .false.
   character(len=:), allocatable, save :: path

   integer(int64), save :: rate = 1_int64
   integer(int64), save :: last = 0_int64
   integer(int64), save :: loop_start = 0_int64
   !> time spent writing the timing file, excluded from the loop total
   integer(int64), save :: io_ticks = 0_int64
   real(wp), save :: stage_time(nstage) = 0.0_wp

contains

subroutine init
   integer :: length, stat

   initialized = .true.
   call get_environment_variable('XTB_SCC_TIMERS', length=length, status=stat)
   if (stat /= 0 .or. length == 0) return
   allocate(character(len=length) :: path)
   call get_environment_variable('XTB_SCC_TIMERS', path)
   call system_clock(count_rate=rate)
   enabled = .true.
end subroutine init

!> Mark the beginning of the SCC iterator (before the overlap factorization)
subroutine scc_timer_start
   if (.not.initialized) call init
   if (.not.enabled) return
   call system_clock(last)
   loop_start = last
   io_ticks = 0_int64
   stage_time(:) = 0.0_wp
end subroutine scc_timer_start

!> Charge the time since the previous mark to a stage
subroutine scc_timer_lap(stage)
   integer, intent(in) :: stage
   integer(int64) :: now

   if (.not.enabled) return
   call system_clock(now)
   stage_time(stage) = stage_time(stage) + real(now - last, wp) / real(rate, wp)
   last = now
end subroutine scc_timer_lap

!> Write the time since scc_timer_start as the setup (overlap factorization)
subroutine scc_timer_setup(ndim)
   integer, intent(in) :: ndim
   integer(int64) :: now
   integer :: unit

   if (.not.enabled) return
   call system_clock(now)
   open(newunit=unit, file=path, position='append', action='write')
   write(unit, '(a,1x,i0,1x,es12.5)') 'setup', ndim, &
      & real(now - last, wp) / real(rate, wp)
   close(unit)
   call system_clock(last)
   io_ticks = io_ticks + (last - now)
end subroutine scc_timer_setup

!> Write the stage times of one cycle and reset them
subroutine scc_timer_cycle(ndim, iter)
   integer, intent(in) :: ndim
   integer, intent(in) :: iter
   integer(int64) :: now
   integer :: unit

   if (.not.enabled) return
   call system_clock(now)
   open(newunit=unit, file=path, position='append', action='write')
   write(unit, '(a,1x,i0,1x,i0,*(1x,es12.5))') 'cycle', ndim, iter, stage_time
   close(unit)
   stage_time(:) = 0.0_wp
   call system_clock(last)
   io_ticks = io_ticks + (last - now)
end subroutine scc_timer_cycle

!> Write the total wall time of the SCC iterator
subroutine scc_timer_done(ndim, cycles)
   integer, intent(in) :: ndim
   integer, intent(in) :: cycles
   integer(int64) :: now
   integer :: unit

   if (.not.enabled) return
   call system_clock(now)
   open(newunit=unit, file=path, position='append', action='write')
   write(unit, '(a,1x,i0,1x,i0,1x,es12.5)') 'done', ndim, cycles, &
      & real(now - loop_start - io_ticks, wp) / real(rate, wp)
   close(unit)
end subroutine scc_timer_done

end module xtb_scc_timer
