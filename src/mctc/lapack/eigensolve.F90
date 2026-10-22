! This file is part of xtb.
!
! Copyright (C) 2019-2020 Sebastian Ehlert
! Copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

!> Wrapper for eigensolver routines
module xtb_mctc_lapack_eigensolve
   use xtb_mctc_accuracy, only : sp, dp
   use xtb_mctc_blas_level3, only : blas_trsm, blas_trmm, blas_gemm
   use xtb_mctc_lapack_geneigval, only : lapack_sygvd
   use xtb_mctc_lapack_stdeigval, only : lapack_syevd
   use xtb_mctc_lapack_gst, only : lapack_sygst
   use xtb_mctc_lapack_trf, only : mctc_potrf
   use xtb_type_environment, only : TEnvironment
#ifdef USE_CUSOLVER
   use xtb_mctc_global
   use cusolverDn
#endif
   implicit none
   private

   public :: TEigenSolver, init


   !> Column-panel width for upper-triangle congruence A = R^T H R (R = U^-1).
   !> Because standard BLAS has no symmetric R^T H R routine, two full-matrix
   !> blas_trmm calls cost 2*N^3 FLOPs; evaluating only rows 1:jj per column
   !> panel j:jj exploits R^T being lower triangular to compute just the upper
   !> triangle of A in 1*N^3 FLOPs (matching dsygst while using gemm + trmm).
   integer, parameter :: nb = 96


   interface
      pure subroutine dtrtri(uplo, diag, n, a, lda, info)
         import :: dp
         character(len=1), intent(in) :: uplo
         character(len=1), intent(in) :: diag
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(dp), intent(inout) :: a(lda, *)
         integer, intent(out) :: info
      end subroutine dtrtri
      pure subroutine dsytrd(uplo, n, a, lda, d, e, tau, work, lwork, info)
         import :: dp
         character(len=1), intent(in) :: uplo
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(dp), intent(inout) :: a(lda, *)
         real(dp), intent(out) :: d(*)
         real(dp), intent(out) :: e(*)
         real(dp), intent(out) :: tau(*)
         real(dp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(out) :: info
      end subroutine dsytrd
      subroutine dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
         import :: dp
         character(len=1), intent(in) :: side, uplo
         integer, intent(in) :: m, n, lda, ldb, ldc
         real(dp), intent(in) :: alpha, beta
         real(dp), intent(in) :: a(lda, *), b(ldb, *)
         real(dp), intent(inout) :: c(ldc, *)
      end subroutine dsymm
      subroutine dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
         import :: dp
         character(len=1), intent(in) :: uplo, trans
         integer, intent(in) :: n, k, lda, ldc
         real(dp), intent(in) :: alpha, beta
         real(dp), intent(in) :: a(lda, *)
         real(dp), intent(inout) :: c(ldc, *)
      end subroutine dsyrk
      subroutine dpotrf(uplo, n, a, lda, info)
         import :: dp
         character(len=1), intent(in) :: uplo
         integer, intent(in) :: n, lda
         real(dp), intent(inout) :: a(lda, *)
         integer, intent(out) :: info
      end subroutine dpotrf
      subroutine dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork, liwork, info)
         import :: dp
         character(len=1), intent(in) :: jobz, uplo
         integer, intent(in) :: n, lda, lwork, liwork
         real(dp), intent(inout) :: a(lda, *)
         real(dp), intent(out) :: w(*)
         real(dp), intent(inout) :: work(*)
         integer, intent(inout) :: iwork(*)
         integer, intent(out) :: info
      end subroutine dsyevd
      pure subroutine dstedc(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info)
         import :: dp
         character(len=1), intent(in) :: compz
         integer, intent(in) :: n
         integer, intent(in) :: ldz
         real(dp), intent(inout) :: d(*)
         real(dp), intent(inout) :: e(*)
         real(dp), intent(inout) :: z(ldz, *)
         real(dp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(inout) :: iwork(*)
         integer, intent(in) :: liwork
         integer, intent(out) :: info
      end subroutine dstedc
      !> LAPACK's dsytrd ('U') with a vectorized symmetric matrix-vector product (sytrd.c)
      subroutine xtb_dsytrd_upper(n, a, lda, d, e, tau, work) bind(c, name='xtb_dsytrd_upper')
         import :: dp
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(dp), intent(inout) :: a(lda, *)
         real(dp), intent(out) :: d(*)
         real(dp), intent(out) :: e(*)
         real(dp), intent(out) :: tau(*)
         real(dp), intent(inout) :: work(*)
      end subroutine xtb_dsytrd_upper
      !> The same reduction on the full symmetric matrix (sytrd.c)
      subroutine xtb_dsytrd_full(n, a, lda, d, e, tau, work) bind(c, name='xtb_dsytrd_full')
         import :: dp
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(dp), intent(inout) :: a(lda, *)
         real(dp), intent(out) :: d(*)
         real(dp), intent(out) :: e(*)
         real(dp), intent(out) :: tau(*)
         real(dp), intent(inout) :: work(*)
      end subroutine xtb_dsytrd_full
      !> dst(1:n, 1:n) = the symmetric matrix with src's upper triangle (sytrd.c)
      subroutine xtb_dsymmetrize_pad(n, src, lds, dst, ldd) bind(c, name='xtb_dsymmetrize_pad')
         import :: dp
         integer, intent(in) :: n
         integer, intent(in) :: lds
         real(dp), intent(in) :: src(lds, *)
         integer, intent(in) :: ldd
         real(dp), intent(inout) :: dst(ldd, *)
      end subroutine xtb_dsymmetrize_pad
      subroutine xtb_ssytrd_upper(n, a, lda, d, e, tau, work) bind(c, name='xtb_ssytrd_upper')
         import :: sp
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(sp), intent(inout) :: a(lda, *)
         real(sp), intent(out) :: d(*)
         real(sp), intent(out) :: e(*)
         real(sp), intent(out) :: tau(*)
         real(sp), intent(inout) :: work(*)
      end subroutine xtb_ssytrd_upper
      !> dstedc('I', ...) on the caller's OpenMP threads; dstedc itself with one (sytrd.c)
      subroutine xtb_dstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info) bind(c, name='xtb_dstedc')
         import :: dp
         integer, intent(in) :: n, ldz, lwork, liwork
         real(dp), intent(inout) :: d(*), e(*)
         real(dp), intent(out) :: z(ldz, *)
         real(dp), intent(inout) :: work(*)
         integer, intent(inout) :: iwork(*)
         integer, intent(out) :: info
      end subroutine xtb_dstedc
      !> sstedc('I', ...) on the caller's OpenMP threads; sstedc itself with one (sytrd.c)
      subroutine xtb_sstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info) bind(c, name='xtb_sstedc')
         import :: sp
         integer, intent(in) :: n, ldz, lwork, liwork
         real(sp), intent(inout) :: d(*), e(*)
         real(sp), intent(out) :: z(ldz, *)
         real(sp), intent(inout) :: work(*)
         integer, intent(inout) :: iwork(*)
         integer, intent(out) :: info
      end subroutine xtb_sstedc
      !> z = Q z with Q from xtb_dsytrd_upper, as dormtr('L', 'U', 'N') but by gemm (sytrd.c)
      subroutine xtb_dormtr_upper(n, m, a, lda, tau, z, ldz, work) bind(c, name='xtb_dormtr_upper')
         import :: dp
         integer, intent(in) :: n, m, lda, ldz
         real(dp), intent(in) :: a(lda, *)
         real(dp), intent(in) :: tau(*)
         real(dp), intent(inout) :: z(ldz, *)
         real(dp), intent(inout) :: work(*)
      end subroutine xtb_dormtr_upper
      subroutine xtb_sormtr_upper(n, m, a, lda, tau, z, ldz, work) bind(c, name='xtb_sormtr_upper')
         import :: sp
         integer, intent(in) :: n, m, lda, ldz
         real(sp), intent(in) :: a(lda, *)
         real(sp), intent(in) :: tau(*)
         real(sp), intent(inout) :: z(ldz, *)
         real(sp), intent(inout) :: work(*)
      end subroutine xtb_sormtr_upper
      pure subroutine ssytrd(uplo, n, a, lda, d, e, tau, work, lwork, info)
         import :: sp
         character(len=1), intent(in) :: uplo
         integer, intent(in) :: n
         integer, intent(in) :: lda
         real(sp), intent(inout) :: a(lda, *)
         real(sp), intent(out) :: d(*)
         real(sp), intent(out) :: e(*)
         real(sp), intent(out) :: tau(*)
         real(sp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(out) :: info
      end subroutine ssytrd
      pure subroutine sstedc(compz, n, d, e, z, ldz, work, lwork, iwork, liwork, info)
         import :: sp
         character(len=1), intent(in) :: compz
         integer, intent(in) :: n
         integer, intent(in) :: ldz
         real(sp), intent(inout) :: d(*)
         real(sp), intent(inout) :: e(*)
         real(sp), intent(inout) :: z(ldz, *)
         real(sp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(inout) :: iwork(*)
         integer, intent(in) :: liwork
         integer, intent(out) :: info
      end subroutine sstedc
      pure subroutine sormtr(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info)
         import :: sp
         character(len=1), intent(in) :: side
         character(len=1), intent(in) :: uplo
         character(len=1), intent(in) :: trans
         integer, intent(in) :: m
         integer, intent(in) :: n
         integer, intent(in) :: lda
         integer, intent(in) :: ldc
         real(sp), intent(in) :: a(lda, *)
         real(sp), intent(in) :: tau(*)
         real(sp), intent(inout) :: c(ldc, *)
         real(sp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(out) :: info
      end subroutine sormtr
      pure subroutine dormtr(side, uplo, trans, m, n, a, lda, tau, c, ldc, work, lwork, info)
         import :: dp
         character(len=1), intent(in) :: side
         character(len=1), intent(in) :: uplo
         character(len=1), intent(in) :: trans
         integer, intent(in) :: m
         integer, intent(in) :: n
         integer, intent(in) :: lda
         integer, intent(in) :: ldc
         real(dp), intent(in) :: a(lda, *)
         real(dp), intent(in) :: tau(*)
         real(dp), intent(inout) :: c(ldc, *)
         real(dp), intent(inout) :: work(*)
         integer, intent(in) :: lwork
         integer, intent(out) :: info
      end subroutine dormtr
   end interface


   integer, parameter :: trd_sym = 0, trd_copy = 1, trd_full = 2

   type :: TEigenSolver
      private
      integer :: n
      integer, allocatable :: iwork(:)
      real(sp), allocatable :: swork(:)
      real(sp), allocatable :: sbmat(:, :)
      real(dp), allocatable :: dwork(:)
      real(dp), allocatable :: dbmat(:, :)
      real(dp), allocatable :: zmat(:, :)
      real(dp), allocatable :: tau(:)
      real(dp), allocatable :: esub(:)
      real(dp), allocatable :: dwork_trd(:)
      !> Workspaces of xtb_dormtr_upper / xtb_sormtr_upper, 32 * (4n + 32) (allocated on first use)
      real(dp), allocatable :: dwork_orm(:)
      real(sp), allocatable :: swork_orm(:)
      !> FP32 copies for mixed-precision SCC cycles (allocated on first use)
      real(sp), allocatable :: amat_s(:, :)
      real(sp), allocatable :: zmat_s(:, :)
      real(sp), allocatable :: tau_s(:)
      real(sp), allocatable :: esub_s(:)
      real(sp), allocatable :: eval_s(:)
      real(sp), allocatable :: swork_dc(:)
      real(sp), allocatable :: swork_trd(:)
      !> Whether the last subspace solve ran in FP32 (selects the back-transformation)
      logical :: last_sp = .false.
      !> Whether the last FP32 solve was refined in FP64 (refine_block): zmat then holds every
      !> eigenvector of X^T H X, and the back-transformation is the triangular product only
      logical :: last_refined = .false.
      !> FP64 workspaces of refine_block (allocated on first use)
      real(dp), allocatable :: rhc(:, :), ry(:, :), rg(:, :), rwork(:)
      integer, allocatable :: riwork(:)
      !> Tridiagonal reduction of the FP64 subspace solve (XTB_SCC_TRD): trd_sym (default) on
      !> the upper triangle; trd_full on a full, padded copy (apad, leading dimension ldp,
      !> 64-byte multiple), 2-12% faster per single point for one process but 2-17% less
      !> throughput with 8 processes x 1 thread (its Accelerate kernels contend; the copy
      !> alone doesn't); trd_copy makes the copy but reduces as trd_sym does
      integer :: trd_mode = 0
      integer :: ldp = 0
      real(dp), allocatable :: apad(:, :)
      !> Whether the last FP64 subspace solve used apad (selects the back-transformation)
      logical :: last_full = .false.
#ifdef USE_CUSOLVER
      integer :: lwork
#endif
   contains
      generic :: solve => sgen_solve, dgen_solve
      procedure :: sgen_solve => mctc_ssygvd
      procedure :: dgen_solve => mctc_dsygvd
      generic :: fact_solve => sfact_solve, dfact_solve
      procedure :: sfact_solve => mctc_ssygvd_factorized
      procedure :: dfact_solve => mctc_dsygvd_factorized
      procedure :: fact_solve_subspace => mctc_dsygvd_fact_subspace
      procedure :: fact_backtransform => mctc_dsygvd_fact_backtransform
      procedure :: fact_solve_subspace_mixed => mctc_dsygvd_fact_subspace_mixed
   end type TEigenSolver


   interface init
      module procedure :: initSEigenSolver
      module procedure :: initDEigenSolver
   end interface init


contains


subroutine initSEigenSolver(self, env, bmat)
   character(len=*), parameter :: source = 'mctc_lapack_sygvd'
   class(TEigenSolver), intent(out) :: self
   type(TEnvironment), intent(inout) :: env
   real(sp), intent(in) :: bmat(:, :)

   self%n = size(bmat, 1)

   allocate(self%swork(1 + 6*self%n + 2*self%n**2))
   allocate(self%iwork(3 + 5*self%n))

   self%sbmat = bmat
   ! Check for Cholesky factorisation
   call mctc_potrf(env, self%sbmat)

end subroutine initSEigenSolver


subroutine initDEigenSolver(self, env, bmat)
   character(len=*), parameter :: source = 'mctc_lapack_sygvd'
   class(TEigenSolver), intent(out) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(in) :: bmat(:, :)
#ifdef USE_CUSOLVER
   integer :: istat, lwork
   ! dummy is only a dummy argument used to query the workspace size needed
   ! for cuSolverDnDsygvd -- it is okay to pass an empty array to cuSolverDnDsygvd_bufferSize
   real(dp) :: dummy(:) 
#endif
   integer :: info, ldwork_cpu, lwork_trd, lda
   real(dp) :: work_trd_q(1), work_orm_q(1)
   logical :: exitRun

   self%n = size(bmat, 1)
   ldwork_cpu = 1 + 6*self%n + 2*self%n**2

#ifdef USE_CUSOLVER
   istat = cusolverDnDsygvd_bufferSize(cusolverDnH, CUSOLVER_EIG_TYPE_1, &
     CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, self%n, dummy,    &
     self%n, dummy, self%n, dummy, lwork)
   if (istat /= 0) then
      call env%error("failed to get dygvd buffer size", source)
   end if

   self%lwork = lwork
   allocate(self%dwork(max(lwork, ldwork_cpu)))
#else
   allocate(self%dwork(ldwork_cpu))
#endif
   allocate(self%iwork(3 + 5*self%n))
   allocate(self%zmat(self%n, self%n), source=0.0_dp)
   allocate(self%tau(self%n), self%esub(self%n))

   self%dbmat = bmat
   lda = max(1, self%n)
   call dsytrd('U', self%n, self%zmat, lda, self%dwork, self%esub, self%tau, &
      & work_trd_q, -1, info)
   call dormtr('L', 'U', 'N', self%n, self%n, self%dbmat, lda, self%tau, &
      & self%zmat, lda, work_orm_q, -1, info)
   lwork_trd = max(1, 32*self%n, int(work_trd_q(1)), int(work_orm_q(1)))
   allocate(self%dwork_trd(lwork_trd))
   call trd_mode_from_env(self)

   ! Check for Cholesky factorisation
   call mctc_potrf(env, self%dbmat)
   call env%check(exitRun)
   if (exitRun) return
   call dtrtri('U', 'N', self%n, self%dbmat, lda, info)
   if (info /= 0) then
      call env%error("Failed to invert Cholesky factor", source)
   end if

end subroutine initDEigenSolver


subroutine mctc_ssygvd(self, env, amat, bmat, eval)
   character(len=*), parameter :: source = 'mctc_lapack_sygvd'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(sp), intent(inout) :: amat(:, :)
   real(sp), intent(in) :: bmat(:, :)
   real(sp), intent(out) :: eval(:)
   integer :: info, lswork, liwork

   self%sbmat(:, :) = bmat

   lswork = size(self%swork)
   liwork = size(self%iwork)
   call lapack_sygvd(1, 'v', 'u', self%n, amat, self%n, self%sbmat, self%n, eval, &
      & self%swork, lswork, self%iwork, liwork, info)

   if (info /= 0) then
      call env%error("Failed to solve eigenvalue problem", source)
   end if

end subroutine mctc_ssygvd


subroutine mctc_dsygvd(self, env, amat, bmat, eval)
   character(len=*), parameter :: source = 'mctc_lapack_sygvd'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(inout) :: amat(:, :)
   real(dp), intent(in) :: bmat(:, :)
   real(dp), intent(out) :: eval(:)
   integer :: info, ldwork, liwork
#ifdef USE_CUSOLVER
   integer :: istat
#endif

   self%dbmat(:, :) = bmat

#ifdef USE_CUSOLVER
   !$acc enter data copyin(amat, self%dbmat, eval, info) create(self%dwork)

   !$acc host_data use_device(amat, self%dbmat, eval, self%dwork, info)
   istat = cusolverDnDsygvd(cusolverDnH, CUSOLVER_EIG_TYPE_1, &
     CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, self%n, amat, self%n, &
     self%dbmat, self%n, eval, self%dwork, self%lwork, info)
   !$acc end host_data

   !$acc exit data copyout(amat, self%dbmat, eval, info) delete(self%dwork)

   if (istat /= 0) then
      call env%error("cuSovlerDnDsygvd failed", source)
   end if
#else
   ldwork = size(self%dwork)
   liwork = size(self%iwork)
   call lapack_sygvd(1, 'v', 'u', self%n, amat, self%n, self%dbmat, self%n, eval, &
      & self%dwork, ldwork, self%iwork, liwork, info)
#endif

   if (info /= 0) then
      call env%error("Failed to solve eigenvalue problem", source)
   end if

end subroutine mctc_dsygvd


subroutine mctc_ssygvd_factorized(self, env, amat, bmat_factorized, eval)
   character(len=*), parameter :: source = 'mctc_lapack_ssygvd_factorized'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(sp), intent(inout) :: amat(:, :)
   real(sp), intent(in) :: bmat_factorized(:, :)
   real(sp), intent(out) :: eval(:)
   integer :: info, lswork, liwork

   lswork = size(self%swork)
   liwork = size(self%iwork)

   CALL lapack_sygst( 1, 'u', self%n, amat, self%n, bmat_factorized, self%n, info )

   if (info /= 0) then
      call env%error("Failed to reduce eigenvalue problem", source)
      return
   end if

   CALL lapack_syevd( 'v', 'u', self%n, amat, self%n, eval, self%swork, lswork, self%iwork, liwork, info )

   if (info /= 0) then
      call env%error("Failed to compute eigenvalues and eigenvectors", source)
      return
   end if

   CALL blas_trsm( 'l', 'u', 'n', 'n', self%n, self%n, 1.0_sp, bmat_factorized, self%n, amat, self%n )

end subroutine mctc_ssygvd_factorized


subroutine mctc_dsygvd_factorized(self, env, amat, bmat_factorized, eval)
   character(len=*), parameter :: source = 'mctc_lapack_dsygvd_factorized'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(inout) :: amat(:, :)
   real(dp), intent(in) :: bmat_factorized(:, :)
   real(dp), intent(out) :: eval(:)
   integer :: info, ldwork, liwork

   ldwork = size(self%dwork)
   liwork = size(self%iwork)

   CALL lapack_sygst( 1, 'u', self%n, amat, self%n, bmat_factorized, self%n, info )

   if (info /= 0) then
      call env%error("Failed to reduce eigenvalue problem", source)
      return
   end if

   CALL lapack_syevd( 'v', 'u', self%n, amat, self%n, eval, self%dwork, ldwork, self%iwork, liwork, info )

   if (info /= 0) then
      call env%error("Failed to compute eigenvalues and eigenvectors", source)
      return
   end if

   CALL blas_trsm( 'l', 'u', 'n', 'n', self%n, self%n, 1.0_dp, bmat_factorized, self%n, amat, self%n )

end subroutine mctc_dsygvd_factorized


subroutine mctc_dsygvd_fact_subspace(self, env, amat, eval, cmat, m_sub)
   character(len=*), parameter :: source = 'mctc_lapack_dsygvd_fact_subspace'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(inout) :: amat(:, :)
   real(dp), intent(out) :: eval(:)
   real(dp), intent(inout), optional :: cmat(:, :)
   integer, intent(in), optional :: m_sub
   integer :: info, ldwork, liwork, ldwork_trd

   ldwork_trd = size(self%dwork_trd)
   ldwork = size(self%dwork)
   liwork = size(self%iwork)

   call congruence(self, amat)
   self%last_sp = .false.
   self%last_refined = .false.
   self%last_full = .false.

   if (self%trd_mode /= trd_sym) then
      if (.not. allocated(self%apad)) then
         self%ldp = (self%n + 7) / 8 * 8
         allocate(self%apad(self%ldp, self%n))
      end if
      call xtb_dsymmetrize_pad(self%n, amat, self%n, self%apad, self%ldp)
   end if
   if (self%trd_mode == trd_full) then
      call xtb_dsytrd_full(self%n, self%apad, self%ldp, eval, self%esub, self%tau, &
         & self%dwork_trd)
      self%last_full = .true.
   else
      call xtb_dsytrd_upper(self%n, amat, self%n, eval, self%esub, self%tau, self%dwork_trd)
   end if
   info = 0
   if (info == 0) then
      call xtb_dstedc(self%n, eval, self%esub, self%zmat, self%n, &
         & self%dwork, ldwork, self%iwork, liwork, info)
   end if
   if (info /= 0) then
      call env%error("Failed to compute eigenvalues and eigenvectors", source)
      return
   end if

   if (present(m_sub) .and. present(cmat)) then
      if (m_sub > 0) then
         call self%fact_backtransform(env, amat, cmat, 1, m_sub)
      end if
   end if

end subroutine mctc_dsygvd_fact_subspace


subroutine mctc_dsygvd_fact_backtransform(self, env, amat, cmat, ilo, ihi)
   character(len=*), parameter :: source = 'mctc_lapack_dsygvd_fact_backtransform'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(in) :: amat(:, :)
   real(dp), intent(inout) :: cmat(:, :)
   integer, intent(in) :: ilo, ihi
   integer :: m

   m = ihi - ilo + 1
   if (m <= 0) return

   if (self%last_refined) then
      ! eigenvectors of X^T H X already formed (FP32 solve refined in FP64)
      call blas_trmm('L', 'U', 'N', 'N', self%n, m, 1.0_dp, self%dbmat, self%n, &
         & self%zmat(:, ilo:ihi), self%n)
      cmat(:, ilo:ihi) = self%zmat(:, ilo:ihi)
      return
   end if

   if (self%last_sp) then
      ! eigenvectors of the FP32 tridiagonal reduction, transformed in FP32
      if (.not. allocated(self%swork_orm)) allocate(self%swork_orm(32 * (4 * self%n + 32)))
      call xtb_sormtr_upper(self%n, m, self%amat_s, self%n, self%tau_s, &
         & self%zmat_s(:, ilo:ihi), self%n, self%swork_orm)
      self%zmat(:, ilo:ihi) = real(self%zmat_s(:, ilo:ihi), dp)
      call blas_trmm('L', 'U', 'N', 'N', self%n, m, 1.0_dp, self%dbmat, self%n, &
         & self%zmat(:, ilo:ihi), self%n)
      cmat(:, ilo:ihi) = self%zmat(:, ilo:ihi)
      return
   end if

   if (.not. allocated(self%dwork_orm)) allocate(self%dwork_orm(32 * (4 * self%n + 32)))
   if (self%last_full) then
      call xtb_dormtr_upper(self%n, m, self%apad, self%ldp, self%tau, &
         & self%zmat(:, ilo:ihi), self%n, self%dwork_orm)
   else
      call xtb_dormtr_upper(self%n, m, amat, self%n, self%tau, &
         & self%zmat(:, ilo:ihi), self%n, self%dwork_orm)
   end if
   call blas_trmm('L', 'U', 'N', 'N', self%n, m, 1.0_dp, self%dbmat, self%n, &
      & self%zmat(:, ilo:ihi), self%n)
   cmat(:, ilo:ihi) = self%zmat(:, ilo:ihi)

end subroutine mctc_dsygvd_fact_backtransform


!> XTB_SCC_TRD = sym | copy | full (default sym; anything else keeps the default)
subroutine trd_mode_from_env(self)
   class(TEigenSolver), intent(inout) :: self
   character(len=16) :: val
   integer :: length, stat

   self%trd_mode = trd_sym
   call get_environment_variable('XTB_SCC_TRD', val, length=length, status=stat)
   if (stat /= 0 .or. length == 0) return
   select case (trim(val))
   case ('copy')
      self%trd_mode = trd_copy
   case ('full')
      self%trd_mode = trd_full
   end select

end subroutine trd_mode_from_env


!> Upper triangle of the congruence X^T H X (X = U^-1 in dbmat), overwriting amat
subroutine congruence(self, amat)
   class(TEigenSolver), intent(inout) :: self
   real(dp), intent(inout) :: amat(:, :)
   integer :: j, jj, k

   do j = 1, self%n, nb
      jj = min(self%n, j + nb - 1)
      k = jj - j + 1
      self%zmat(1:jj, j:jj) = amat(1:jj, j:jj)
      call blas_trmm('R', 'U', 'N', 'N', jj, k, 1.0_dp, self%dbmat(j:jj, j:jj), k, &
         & self%zmat(:, j:jj), self%n)
      call blas_gemm('N', 'N', jj, k, j - 1, 1.0_dp, amat, self%n, &
         & self%dbmat(:, j:jj), self%n, 1.0_dp, self%zmat(:, j:jj), self%n)
      call blas_trmm('L', 'U', 'T', 'N', jj, k, 1.0_dp, self%dbmat, self%n, &
         & self%zmat(:, j:jj), self%n)
   end do
   do j = 1, self%n
      amat(1:j, j) = self%zmat(1:j, j)
   end do

end subroutine congruence


!> As mctc_dsygvd_fact_subspace, but the tridiagonal reduction, divide and conquer
!> and the later back-transformation (fact_backtransform) run in FP32. Meant for the
!> early SCC cycles only: eigenvalues and density carry FP32 errors (~1e-6 relative).
!> With nref > 0, the FP32 back-transformation forms every eigenvector and refine_block
!> corrects the lowest nref (or more) in FP64, which removes those errors for them.
!> With stat present, a failure (LAPACK info, nonzero) is returned there instead of raised as
!> an error, so the caller can redo the work in FP64. Accelerate's legacy sstedc fails on rare
!> matrices that dstedc solves: its slaed4 doesn't converge on one root of a merge.
subroutine mctc_dsygvd_fact_subspace_mixed(self, env, amat, eval, nref, stat)
   character(len=*), parameter :: source = 'mctc_lapack_dsygvd_fact_subspace_mixed'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(inout) :: amat(:, :)
   real(dp), intent(out) :: eval(:)
   integer, intent(in), optional :: nref
   integer, intent(out), optional :: stat
   integer :: j, info
   real(sp) :: work_trd_q(1), work_orm_q(1)

   if (present(stat)) stat = 0

   if (.not. allocated(self%amat_s)) then
      allocate(self%amat_s(self%n, self%n), self%zmat_s(self%n, self%n), &
         & self%tau_s(self%n), self%esub_s(self%n), self%eval_s(self%n), &
         & self%swork_dc(1 + 4*self%n + self%n**2))
      call ssytrd('U', self%n, self%amat_s, self%n, self%eval_s, self%esub_s, &
         & self%tau_s, work_trd_q, -1, info)
      call sormtr('L', 'U', 'N', self%n, self%n, self%amat_s, self%n, self%tau_s, &
         & self%zmat_s, self%n, work_orm_q, -1, info)
      allocate(self%swork_trd(max(1, 32*self%n, int(work_trd_q(1)), int(work_orm_q(1)))))
   end if

   call congruence(self, amat)
   do j = 1, self%n
      self%amat_s(1:j, j) = real(amat(1:j, j), sp)
   end do

   call xtb_ssytrd_upper(self%n, self%amat_s, self%n, self%eval_s, self%esub_s, self%tau_s, &
      & self%swork_trd)
   info = 0
   if (info == 0) then
      call xtb_sstedc(self%n, self%eval_s, self%esub_s, self%zmat_s, self%n, &
         & self%swork_dc, size(self%swork_dc), self%iwork, size(self%iwork), info)
   end if
   if (info /= 0) then
      self%last_sp = .false.
      self%last_refined = .false.
      if (present(stat)) then
         stat = info
      else
         call env%error("Failed to compute eigenvalues and eigenvectors", source)
      end if
      return
   end if
   eval = real(self%eval_s, dp)
   self%last_sp = .true.
   self%last_refined = .false.

   if (present(nref)) then
      if (nref > 0) then
         if (.not. allocated(self%swork_orm)) allocate(self%swork_orm(32 * (4 * self%n + 32)))
         call xtb_sormtr_upper(self%n, self%n, self%amat_s, self%n, self%tau_s, &
            & self%zmat_s, self%n, self%swork_orm)
         self%zmat(:, :) = real(self%zmat_s, dp)
         call refine_block(self, env, amat, eval, nref, stat)
      end if
   end if

end subroutine mctc_dsygvd_fact_subspace_mixed


!> FP64 correction of the lowest k eigenpairs of A = X^T H X (upper triangle in amat) from
!> FP32 ones (all n vectors in zmat, eigenvalues in eval). k = nref, raised until the gap
!> to the next eigenvalue is at least 1e-3 Eh. One first-order step of each z_i against all
!> z_j, j /= i, z_i -= sum_j z_j y_ji / (e_j - e_i) with y_ji = z_j^T (A z_i - e_i z_i) (the
!> residual form cancels the FP32 vectors' own non-orthogonality), skipping pairs closer
!> than 1e-3 Eh; eigenvalues from the Rayleigh quotients e_i + z_i^T r_i / z_i^T z_i of the
!> FP32 vectors (second order); Cholesky QR; then Rayleigh-Ritz on each cluster of the
!> skipped pairs (consecutive eigenvalues closer than 1e-3 Eh, at most a few orbitals).
!> About 6 n^2 k flops. On dumped GFN2 matrices (scripts/kernels/fp32_error.c) the occupied
!> subspace goes from ~2e-6 (FP32) to ~1e-9 (sine of the largest principal angle against
!> FP64), eigenvalues from ~1e-5 to <= 5e-9 Eh. A full Rayleigh-Ritz on the block reaches
!> 1e-11 but its k x k eigensolve costs as much as FP32 saves (scripts/kernels/refine_cost.c).
!> With stat present, a failure is returned there (LAPACK info) instead of raised as an error.
subroutine refine_block(self, env, amat, eval, nref, stat)
   character(len=*), parameter :: source = 'mctc_lapack_refine_block'
   class(TEigenSolver), intent(inout) :: self
   type(TEnvironment), intent(inout) :: env
   real(dp), intent(in) :: amat(:, :)
   real(dp), intent(inout) :: eval(:)
   integer, intent(in) :: nref
   integer, intent(inout), optional :: stat
   real(dp), parameter :: gap_min = 1.0e-3_dp
   integer :: n, k, i, j, i0, i1, c, info, lw, liw
   real(dp) :: g, zz, zr
   real(dp) :: lw_q(1)
   integer :: liw_q(1)

   n = self%n
   k = min(n, max(1, nref))
   do while (k < n)
      if (eval(k + 1) - eval(k) >= gap_min) exit
      k = k + 1
   end do
   if (.not. allocated(self%rhc)) then
      allocate(self%rhc(n, n), self%ry(n, n), self%rg(n, n))
      call dsyevd('V', 'U', n, self%rg, n, eval, lw_q, -1, liw_q, -1, info)
      allocate(self%rwork(max(1, int(lw_q(1)))), self%riwork(max(1, liw_q(1))))
   end if
   lw = size(self%rwork)
   liw = size(self%riwork)

   ! residual R = A Z_k - Z_k diag(e) and the Rayleigh quotients (kept in rg(:, 1))
   call dsymm('L', 'U', n, k, 1.0_dp, amat, n, self%zmat, n, 0.0_dp, self%rhc, n)
   do i = 1, k
      self%rhc(:, i) = self%rhc(:, i) - eval(i) * self%zmat(:, i)
      zz = dot_product(self%zmat(:, i), self%zmat(:, i))
      zr = dot_product(self%zmat(:, i), self%rhc(:, i))
      self%rg(i, 1) = eval(i) + zr / zz
   end do
   ! Y = Z^T R, scaled by the gaps; Z_k -= Z Y
   call blas_gemm('T', 'N', n, k, n, 1.0_dp, self%zmat, n, self%rhc, n, 0.0_dp, self%ry, n)
   do i = 1, k
      do j = 1, n
         g = eval(j) - eval(i)
         if (j == i .or. abs(g) < gap_min) then
            self%ry(j, i) = 0.0_dp
         else
            self%ry(j, i) = self%ry(j, i) / g
         end if
      end do
   end do
   self%rhc(:, 1:k) = self%zmat(:, 1:k)
   call blas_gemm('N', 'N', n, k, n, -1.0_dp, self%zmat, n, self%ry, n, 1.0_dp, self%rhc, n)
   self%zmat(:, 1:k) = self%rhc(:, 1:k)
   eval(1:k) = self%rg(1:k, 1)
   ! Cholesky QR of Z_k
   call dsyrk('U', 'T', k, n, 1.0_dp, self%zmat, n, 0.0_dp, self%rg, n)
   call dpotrf('U', k, self%rg, n, info)
   if (info /= 0) then
      self%last_refined = .false.
      if (present(stat)) then
         stat = info
      else
         call env%error("Cholesky QR failed in the FP64 refinement", source)
      end if
      return
   end if
   call blas_trsm('R', 'U', 'N', 'N', n, k, 1.0_dp, self%rg, n, self%zmat, n)
   ! Rayleigh-Ritz on each cluster of consecutive eigenvalues closer than gap_min
   i0 = 1
   do while (i0 <= k)
      i1 = i0
      do while (i1 < k)
         if (eval(i1 + 1) - eval(i1) >= gap_min) exit
         i1 = i1 + 1
      end do
      c = i1 - i0 + 1
      if (c > 1) then
         call dsymm('L', 'U', n, c, 1.0_dp, amat, n, self%zmat(:, i0:i1), n, 0.0_dp, &
            & self%rhc, n)
         call blas_gemm('T', 'N', c, c, n, 1.0_dp, self%zmat(:, i0:i1), n, self%rhc, n, &
            & 0.0_dp, self%rg, n)
         call dsyevd('V', 'U', c, self%rg, n, eval(i0:i1), self%rwork, lw, self%riwork, &
            & liw, info)
         if (info /= 0) then
            self%last_refined = .false.
            if (present(stat)) then
               stat = info
            else
               call env%error("Rayleigh-Ritz eigensolve failed in the FP64 refinement", source)
            end if
            return
         end if
         call blas_gemm('N', 'N', n, c, c, 1.0_dp, self%zmat(:, i0:i1), n, self%rg, n, &
            & 0.0_dp, self%rhc, n)
         self%zmat(:, i0:i1) = self%rhc(:, 1:c)
      end if
      i0 = i1 + 1
   end do
   self%last_sp = .false.
   self%last_refined = .true.

end subroutine refine_block

end module xtb_mctc_lapack_eigensolve
