/* This file is part of xtb.
 *
 * xtb is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * xtb is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with xtb.  If not, see <https://www.gnu.org/licenses/>.
 */

/* Reduction of a symmetric matrix (upper triangle) to tridiagonal form: LAPACK's
 * blocked dsytrd (dlatrd panels of 32 columns + dsyr2k, dsytd2 for the last block),
 * ported unchanged, so the output (Householder vectors in A, tau) is what dormtr
 * expects. The only change: on aarch64 the symmetric matrix-vector product, the
 * BLAS2 core of the reduction, is a hand-vectorized kernel that reads the triangle
 * once for eight columns at a time (~55-65 GB/s on M1 Max against ~30 GB/s for
 * Accelerate's dsymv; dsytrd 30% faster at n = 412-776). Elsewhere it calls dsymv.
 *
 * xtb_dsytrd_full is the same reduction on the full matrix (both triangles stored and
 * updated): dgemv 'T' (A^T x = A x) instead of the symmetric product, two dgemm instead of
 * dsyr2k, two dger instead of dsyr2. Twice the arithmetic, but Accelerate's full-matrix
 * kernels are fast enough that with a leading dimension padded to 64 bytes it is 2-19%
 * faster at n = 412-776 (dgemv 'T' reads the matrix at 100-155 GB/s, the symmetric product
 * at ~50). xtb_dsymmetrize_pad builds its input from the upper triangle. */

#include <stddef.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef int lint;
extern void dgemv_(const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, const lint *, const double *, double *, const lint *);
extern void dsymv_(const char *, const lint *, const double *, const double *, const lint *, const double *,
                   const lint *, const double *, double *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *,
                    const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsyr2_(const char *, const lint *, const double *, const double *, const lint *, const double *,
                   const lint *, double *, const lint *);
extern void dlarfg_(const lint *, double *, double *, const lint *, double *);
extern double ddot_(const lint *, const double *, const lint *, const double *, const lint *);
extern void daxpy_(const lint *, const double *, const double *, const lint *, double *, const lint *);
extern void dscal_(const lint *, const double *, double *, const lint *);
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *,
                   const double *, const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dger_(const lint *, const lint *, const double *, const double *, const lint *, const double *,
                  const lint *, double *, const lint *);
extern void sgemv_(const char *, const lint *, const lint *, const float *, const float *, const lint *,
                   const float *, const lint *, const float *, float *, const lint *);
extern void ssymv_(const char *, const lint *, const float *, const float *, const lint *, const float *,
                   const lint *, const float *, float *, const lint *);
extern void ssyr2k_(const char *, const char *, const lint *, const lint *, const float *, const float *,
                    const lint *, const float *, const lint *, const float *, float *, const lint *);
extern void ssyr2_(const char *, const lint *, const float *, const float *, const lint *, const float *,
                   const lint *, float *, const lint *);
extern void slarfg_(const lint *, float *, float *, const lint *, float *);
extern void saxpy_(const lint *, const float *, const float *, const lint *, float *, const lint *);
extern void sscal_(const lint *, const float *, float *, const lint *);

#if defined(__aarch64__)
// y = alpha * A x, A symmetric, upper triangle, n x n, column-major with lda
static void symv_upper(int n, double alpha, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0;
  int j0 = 0;
  // Eight columns per pass while they fit: each column's dot product is a chain of one fused multiply-add per
  // vector of rows (4-cycle latency), so eight chains keep the four FMA pipes busy where four can't (1.1-1.25x).
  // Same bits as two passes of four: rows above j0 see the eight columns in the same order, and the last four
  // columns' rows j0 .. j0 + 3 follow the first four's diagonal block, as they would.
  for (; j0 + 8 <= n; j0 += 8) {
    const double *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda, *c4 = c3 + lda,
              *c5 = c4 + lda, *c6 = c5 + lda, *c7 = c6 + lda;
    float64x2_t x0 = vdupq_n_f64(x[j0]), x1 = vdupq_n_f64(x[j0 + 1]), x2 = vdupq_n_f64(x[j0 + 2]),
                x3 = vdupq_n_f64(x[j0 + 3]), x4 = vdupq_n_f64(x[j0 + 4]), x5 = vdupq_n_f64(x[j0 + 5]),
                x6 = vdupq_n_f64(x[j0 + 6]), x7 = vdupq_n_f64(x[j0 + 7]);
    float64x2_t s0 = vdupq_n_f64(0), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    int i = 0;
    for (; i + 2 <= j0; i += 2) {
      float64x2_t a0 = vld1q_f64(c0 + i), a1 = vld1q_f64(c1 + i), a2 = vld1q_f64(c2 + i), a3 = vld1q_f64(c3 + i);
      float64x2_t a4 = vld1q_f64(c4 + i), a5 = vld1q_f64(c5 + i), a6 = vld1q_f64(c6 + i), a7 = vld1q_f64(c7 + i);
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      yi = vfmaq_f64(yi, a0, x0); yi = vfmaq_f64(yi, a1, x1); yi = vfmaq_f64(yi, a2, x2); yi = vfmaq_f64(yi, a3, x3);
      yi = vfmaq_f64(yi, a4, x4); yi = vfmaq_f64(yi, a5, x5); yi = vfmaq_f64(yi, a6, x6); yi = vfmaq_f64(yi, a7, x7);
      vst1q_f64(y + i, yi);
      s0 = vfmaq_f64(s0, a0, xi); s1 = vfmaq_f64(s1, a1, xi); s2 = vfmaq_f64(s2, a2, xi); s3 = vfmaq_f64(s3, a3, xi);
      s4 = vfmaq_f64(s4, a4, xi); s5 = vfmaq_f64(s5, a5, xi); s6 = vfmaq_f64(s6, a6, xi); s7 = vfmaq_f64(s7, a7, xi);
    }
    y[j0] += vaddvq_f64(s0); y[j0 + 1] += vaddvq_f64(s1); y[j0 + 2] += vaddvq_f64(s2); y[j0 + 3] += vaddvq_f64(s3);
    for (int c = 0; c < 4; c++) {
      const double *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
    for (; i + 2 <= j0 + 4; i += 2) {
      float64x2_t a4 = vld1q_f64(c4 + i), a5 = vld1q_f64(c5 + i), a6 = vld1q_f64(c6 + i), a7 = vld1q_f64(c7 + i);
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      yi = vfmaq_f64(yi, a4, x4); yi = vfmaq_f64(yi, a5, x5); yi = vfmaq_f64(yi, a6, x6); yi = vfmaq_f64(yi, a7, x7);
      vst1q_f64(y + i, yi);
      s4 = vfmaq_f64(s4, a4, xi); s5 = vfmaq_f64(s5, a5, xi); s6 = vfmaq_f64(s6, a6, xi); s7 = vfmaq_f64(s7, a7, xi);
    }
    y[j0 + 4] += vaddvq_f64(s4); y[j0 + 5] += vaddvq_f64(s5); y[j0 + 6] += vaddvq_f64(s6); y[j0 + 7] += vaddvq_f64(s7);
    for (int c = 4; c < 8; c++) {
      const double *col = A + (size_t)(j0 + c) * lda;
      for (int r = 4; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 + 4 <= n; j0 += 4) {
    const double *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    float64x2_t x0 = vdupq_n_f64(x[j0]), x1 = vdupq_n_f64(x[j0 + 1]), x2 = vdupq_n_f64(x[j0 + 2]),
                x3 = vdupq_n_f64(x[j0 + 3]);
    float64x2_t s0 = vdupq_n_f64(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 2 <= j0; i += 2) {
      float64x2_t a0 = vld1q_f64(c0 + i), a1 = vld1q_f64(c1 + i), a2 = vld1q_f64(c2 + i), a3 = vld1q_f64(c3 + i);
      float64x2_t xi = vld1q_f64(x + i), yi = vld1q_f64(y + i);
      yi = vfmaq_f64(yi, a0, x0); yi = vfmaq_f64(yi, a1, x1); yi = vfmaq_f64(yi, a2, x2); yi = vfmaq_f64(yi, a3, x3);
      vst1q_f64(y + i, yi);
      s0 = vfmaq_f64(s0, a0, xi); s1 = vfmaq_f64(s1, a1, xi); s2 = vfmaq_f64(s2, a2, xi); s3 = vfmaq_f64(s3, a3, xi);
    }
    double t0 = vaddvq_f64(s0), t1 = vaddvq_f64(s1), t2 = vaddvq_f64(s2), t3 = vaddvq_f64(s3);
    for (; i < j0; i++) {
      y[i] += c0[i] * x[j0] + c1[i] * x[j0 + 1] + c2[i] * x[j0 + 2] + c3[i] * x[j0 + 3];
      t0 += c0[i] * x[i]; t1 += c1[i] * x[i]; t2 += c2[i] * x[i]; t3 += c3[i] * x[i];
    }
    y[j0] += t0; y[j0 + 1] += t1; y[j0 + 2] += t2; y[j0 + 3] += t3;
    for (int c = 0; c < 4; c++) {
      const double *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 < n; j0++) {
    const double *col = A + (size_t)j0 * lda;
    double s = 0;
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }
    y[j0] += s + col[j0] * x[j0];
  }
  if (alpha != 1.0) for (int i = 0; i < n; i++) y[i] *= alpha;
}
#endif

static void symv(lint n, double alpha, const double *A, lint lda, const double *x, double *y) {
#if defined(__aarch64__)
  symv_upper(n, alpha, A, lda, x, y);
#else
  double zero = 0; lint one = 1;
  dsymv_("U", &n, &alpha, A, &lda, x, &one, &zero, y, &one);
#endif
}

// y = alpha A x; sym: upper triangle only, else the full matrix (as A^T x)
static void matvec(int sym, lint n, double alpha, const double *A, lint lda, const double *x, double *y) {
  if (sym) { symv(n, alpha, A, lda, x, y); return; }
  double zero = 0; lint one = 1;
  dgemv_("T", &n, &n, &alpha, A, &lda, x, &one, &zero, y, &one);
}

#define A_(i, j) a[((i) - 1) + (size_t)((j) - 1) * lda]
#define W_(i, j) w[((i) - 1) + (size_t)((j) - 1) * ldw]

// LAPACK dlatrd, UPLO = 'U'
static void latrd_u(int sym, lint n, lint nb, double *a, lint lda, double *e, double *tau, double *w, lint ldw) {
  const double one = 1, mone = -1, zero = 0, half = 0.5; const lint i1 = 1;
  for (lint i = n; i >= n - nb + 1; i--) {
    lint iw = i - n + nb;
    if (i < n) {
      lint m = n - i;
      dgemv_("N", &i, &m, &mone, &A_(1, i + 1), &lda, &W_(i, iw + 1), &ldw, &one, &A_(1, i), &i1);
      dgemv_("N", &i, &m, &mone, &W_(1, iw + 1), &ldw, &A_(i, i + 1), &lda, &one, &A_(1, i), &i1);
    }
    if (i > 1) {
      lint im1 = i - 1;
      dlarfg_(&im1, &A_(i - 1, i), &A_(1, i), &i1, &tau[i - 2]);
      e[i - 2] = A_(i - 1, i);
      A_(i - 1, i) = one;
      matvec(sym, im1, one, a, lda, &A_(1, i), &W_(1, iw));
      if (i < n) {
        lint m = n - i;
        dgemv_("T", &im1, &m, &one, &W_(1, iw + 1), &ldw, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &A_(1, i + 1), &lda, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
        dgemv_("T", &im1, &m, &one, &A_(1, i + 1), &lda, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &W_(1, iw + 1), &ldw, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
      }
      dscal_(&im1, &tau[i - 2], &W_(1, iw), &i1);
      double alpha = -half * tau[i - 2] * ddot_(&im1, &W_(1, iw), &i1, &A_(1, i), &i1);
      daxpy_(&im1, &alpha, &A_(1, i), &i1, &W_(1, iw), &i1);
    }
  }
}

// LAPACK dsytd2, UPLO = 'U'
static void sytd2_u(int sym, lint n, double *a, lint lda, double *d, double *e, double *tau) {
  const double one = 1, mone = -1, half = 0.5; const lint i1 = 1;
  for (lint i = n - 1; i >= 1; i--) {
    double taui;
    dlarfg_(&i, &A_(i, i + 1), &A_(1, i + 1), &i1, &taui);
    e[i - 1] = A_(i, i + 1);
    if (taui != 0) {
      A_(i, i + 1) = one;
      matvec(sym, i, taui, a, lda, &A_(1, i + 1), tau);
      double alpha = -half * taui * ddot_(&i, tau, &i1, &A_(1, i + 1), &i1);
      daxpy_(&i, &alpha, &A_(1, i + 1), &i1, tau, &i1);
      if (sym) dsyr2_("U", &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);
      else {
        dger_(&i, &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);
        dger_(&i, &i, &mone, tau, &i1, &A_(1, i + 1), &i1, a, &lda);
      }
      A_(i, i + 1) = e[i - 1];
    }
    d[i] = A_(i + 1, i + 1);
    tau[i - 1] = taui;
  }
  d[0] = A_(1, 1);
}

// LAPACK dsytrd, UPLO = 'U', nb = 32, crossover nx = 32 (ilaenv defaults); work >= n * nb
static void sytrd_u(int sym, lint n, double *a, lint lda, double *d, double *e, double *tau, double *work) {
  const lint nb = 32, nx = 32; const double mone = -1, one = 1;
  lint kk = 0;
  if (n > nx) {
    kk = n - ((n - nx + nb - 1) / nb) * nb;
    for (lint i = n - nb + 1; i >= kk + 1; i -= nb) {
      lint m = i + nb - 1, im1 = i - 1;
      latrd_u(sym, m, nb, a, lda, e, tau, work, n);
      if (sym) dsyr2k_("U", "N", &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);
      else {
        dgemm_("N", "T", &im1, &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);
        dgemm_("N", "T", &im1, &im1, &nb, &mone, work, &n, &A_(1, i), &lda, &one, a, &lda);
      }
      for (lint j = i; j <= i + nb - 1; j++) { A_(j - 1, j) = e[j - 2]; d[j - 1] = A_(j, j); }
    }
  } else kk = n;
  sytd2_u(sym, kk, a, lda, d, e, tau);
}


/* Fortran: call xtb_dsytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32 */
void xtb_dsytrd_upper(const lint *n, double *a, const lint *lda, double *d, double *e, double *tau, double *work) {
  sytrd_u(1, *n, a, *lda, d, e, tau, work);
}

/* Fortran: call xtb_dsytrd_full(n, a, lda, d, e, tau, work): a holds the full symmetric
 * matrix; on return its upper triangle holds what xtb_dsytrd_upper leaves there (the lower
 * triangle is overwritten). work of size n * 32 */
void xtb_dsytrd_full(const lint *n, double *a, const lint *lda, double *d, double *e, double *tau, double *work) {
  sytrd_u(0, *n, a, *lda, d, e, tau, work);
}

/* Fortran: call xtb_dsymmetrize_pad(n, src, lds, dst, ldd): dst(1:n, 1:n) = the symmetric
 * matrix whose upper triangle is src's (src's lower triangle is not read), in 32 x 32 tiles */
void xtb_dsymmetrize_pad(const lint *np, const double *src, const lint *ldsp, double *dst, const lint *lddp) {
  const lint n = *np, lds = *ldsp, ldd = *lddp, tb = 32;
  for (lint j0 = 0; j0 < n; j0 += tb) {
    lint j1 = j0 + tb < n ? j0 + tb : n;
    for (lint i0 = 0; i0 <= j0; i0 += tb) {
      lint i1 = i0 + tb < n ? i0 + tb : n;
      for (lint j = j0; j < j1; j++) {
        lint iend = i1 < j + 1 ? i1 : j + 1;
        for (lint i = i0; i < iend; i++) {
          double v = src[i + (size_t)j * lds];
          dst[i + (size_t)j * ldd] = v;
          dst[j + (size_t)i * ldd] = v;
        }
      }
    }
  }
}

/* FP32 version for the mixed-precision SCC cycles. Same algorithm; the dot products
 * are computed here because the legacy Fortran ABI of sdot_ (Accelerate) returns a
 * double. */
#if defined(__aarch64__)
static void symv_upper_f(int n, float alpha, const float *A, int lda, const float *x, float *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0f;
  int j0 = 0;
  // Eight columns per pass while they fit: each column's dot product is a chain of one fused multiply-add per
  // vector of rows (4-cycle latency), so eight chains keep the four FMA pipes busy where four can't (1.1-1.25x).
  // Same bits as two passes of four: rows above j0 see the eight columns in the same order, and the last four
  // columns' rows j0 .. j0 + 3 follow the first four's diagonal block, as they would.
  for (; j0 + 8 <= n; j0 += 8) {
    const float *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda, *c4 = c3 + lda,
              *c5 = c4 + lda, *c6 = c5 + lda, *c7 = c6 + lda;
    float32x4_t x0 = vdupq_n_f32(x[j0]), x1 = vdupq_n_f32(x[j0 + 1]), x2 = vdupq_n_f32(x[j0 + 2]),
                x3 = vdupq_n_f32(x[j0 + 3]), x4 = vdupq_n_f32(x[j0 + 4]), x5 = vdupq_n_f32(x[j0 + 5]),
                x6 = vdupq_n_f32(x[j0 + 6]), x7 = vdupq_n_f32(x[j0 + 7]);
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    int i = 0;
    for (; i + 4 <= j0; i += 4) {
      float32x4_t a0 = vld1q_f32(c0 + i), a1 = vld1q_f32(c1 + i), a2 = vld1q_f32(c2 + i), a3 = vld1q_f32(c3 + i);
      float32x4_t a4 = vld1q_f32(c4 + i), a5 = vld1q_f32(c5 + i), a6 = vld1q_f32(c6 + i), a7 = vld1q_f32(c7 + i);
      float32x4_t xi = vld1q_f32(x + i), yi = vld1q_f32(y + i);
      yi = vfmaq_f32(yi, a0, x0); yi = vfmaq_f32(yi, a1, x1); yi = vfmaq_f32(yi, a2, x2); yi = vfmaq_f32(yi, a3, x3);
      yi = vfmaq_f32(yi, a4, x4); yi = vfmaq_f32(yi, a5, x5); yi = vfmaq_f32(yi, a6, x6); yi = vfmaq_f32(yi, a7, x7);
      vst1q_f32(y + i, yi);
      s0 = vfmaq_f32(s0, a0, xi); s1 = vfmaq_f32(s1, a1, xi); s2 = vfmaq_f32(s2, a2, xi); s3 = vfmaq_f32(s3, a3, xi);
      s4 = vfmaq_f32(s4, a4, xi); s5 = vfmaq_f32(s5, a5, xi); s6 = vfmaq_f32(s6, a6, xi); s7 = vfmaq_f32(s7, a7, xi);
    }
    y[j0] += vaddvq_f32(s0); y[j0 + 1] += vaddvq_f32(s1); y[j0 + 2] += vaddvq_f32(s2); y[j0 + 3] += vaddvq_f32(s3);
    for (int c = 0; c < 4; c++) {
      const float *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
    for (; i + 4 <= j0 + 4; i += 4) {
      float32x4_t a4 = vld1q_f32(c4 + i), a5 = vld1q_f32(c5 + i), a6 = vld1q_f32(c6 + i), a7 = vld1q_f32(c7 + i);
      float32x4_t xi = vld1q_f32(x + i), yi = vld1q_f32(y + i);
      yi = vfmaq_f32(yi, a4, x4); yi = vfmaq_f32(yi, a5, x5); yi = vfmaq_f32(yi, a6, x6); yi = vfmaq_f32(yi, a7, x7);
      vst1q_f32(y + i, yi);
      s4 = vfmaq_f32(s4, a4, xi); s5 = vfmaq_f32(s5, a5, xi); s6 = vfmaq_f32(s6, a6, xi); s7 = vfmaq_f32(s7, a7, xi);
    }
    y[j0 + 4] += vaddvq_f32(s4); y[j0 + 5] += vaddvq_f32(s5); y[j0 + 6] += vaddvq_f32(s6); y[j0 + 7] += vaddvq_f32(s7);
    for (int c = 4; c < 8; c++) {
      const float *col = A + (size_t)(j0 + c) * lda;
      for (int r = 4; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 + 4 <= n; j0 += 4) {
    const float *c0 = A + (size_t)j0 * lda, *c1 = c0 + lda, *c2 = c1 + lda, *c3 = c2 + lda;
    float32x4_t x0 = vdupq_n_f32(x[j0]), x1 = vdupq_n_f32(x[j0 + 1]), x2 = vdupq_n_f32(x[j0 + 2]),
                x3 = vdupq_n_f32(x[j0 + 3]);
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 4 <= j0; i += 4) {
      float32x4_t a0 = vld1q_f32(c0 + i), a1 = vld1q_f32(c1 + i), a2 = vld1q_f32(c2 + i), a3 = vld1q_f32(c3 + i);
      float32x4_t xi = vld1q_f32(x + i), yi = vld1q_f32(y + i);
      yi = vfmaq_f32(yi, a0, x0); yi = vfmaq_f32(yi, a1, x1); yi = vfmaq_f32(yi, a2, x2); yi = vfmaq_f32(yi, a3, x3);
      vst1q_f32(y + i, yi);
      s0 = vfmaq_f32(s0, a0, xi); s1 = vfmaq_f32(s1, a1, xi); s2 = vfmaq_f32(s2, a2, xi); s3 = vfmaq_f32(s3, a3, xi);
    }
    float t0 = vaddvq_f32(s0), t1 = vaddvq_f32(s1), t2 = vaddvq_f32(s2), t3 = vaddvq_f32(s3);
    for (; i < j0; i++) {
      y[i] += c0[i] * x[j0] + c1[i] * x[j0 + 1] + c2[i] * x[j0 + 2] + c3[i] * x[j0 + 3];
      t0 += c0[i] * x[i]; t1 += c1[i] * x[i]; t2 += c2[i] * x[i]; t3 += c3[i] * x[i];
    }
    y[j0] += t0; y[j0 + 1] += t1; y[j0 + 2] += t2; y[j0 + 3] += t3;
    for (int c = 0; c < 4; c++) {
      const float *col = A + (size_t)(j0 + c) * lda;
      for (int r = 0; r < c; r++) { y[j0 + r] += col[j0 + r] * x[j0 + c]; y[j0 + c] += col[j0 + r] * x[j0 + r]; }
      y[j0 + c] += col[j0 + c] * x[j0 + c];
    }
  }
  for (; j0 < n; j0++) {
    const float *col = A + (size_t)j0 * lda;
    float s = 0;
    for (int i = 0; i < j0; i++) { y[i] += col[i] * x[j0]; s += col[i] * x[i]; }
    y[j0] += s + col[j0] * x[j0];
  }
  if (alpha != 1.0f) for (int i = 0; i < n; i++) y[i] *= alpha;
}
#endif

static void symv_f(lint n, float alpha, const float *A, lint lda, const float *x, float *y) {
#if defined(__aarch64__)
  symv_upper_f(n, alpha, A, lda, x, y);
#else
  float zero = 0; lint one = 1;
  ssymv_("U", &n, &alpha, A, &lda, x, &one, &zero, y, &one);
#endif
}

static float dot_f(lint n, const float *x, const float *y) {
  float s = 0;
  for (lint i = 0; i < n; i++) s += x[i] * y[i];
  return s;
}

#define AF_(i, j) a[((i) - 1) + (size_t)((j) - 1) * lda]
#define WF_(i, j) w[((i) - 1) + (size_t)((j) - 1) * ldw]

static void latrd_u_f(lint n, lint nb, float *a, lint lda, float *e, float *tau, float *w, lint ldw) {
  const float one = 1, mone = -1, zero = 0, half = 0.5f; const lint i1 = 1;
  for (lint i = n; i >= n - nb + 1; i--) {
    lint iw = i - n + nb;
    if (i < n) {
      lint m = n - i;
      sgemv_("N", &i, &m, &mone, &AF_(1, i + 1), &lda, &WF_(i, iw + 1), &ldw, &one, &AF_(1, i), &i1);
      sgemv_("N", &i, &m, &mone, &WF_(1, iw + 1), &ldw, &AF_(i, i + 1), &lda, &one, &AF_(1, i), &i1);
    }
    if (i > 1) {
      lint im1 = i - 1;
      slarfg_(&im1, &AF_(i - 1, i), &AF_(1, i), &i1, &tau[i - 2]);
      e[i - 2] = AF_(i - 1, i);
      AF_(i - 1, i) = one;
      symv_f(im1, one, a, lda, &AF_(1, i), &WF_(1, iw));
      if (i < n) {
        lint m = n - i;
        sgemv_("T", &im1, &m, &one, &WF_(1, iw + 1), &ldw, &AF_(1, i), &i1, &zero, &WF_(i + 1, iw), &i1);
        sgemv_("N", &im1, &m, &mone, &AF_(1, i + 1), &lda, &WF_(i + 1, iw), &i1, &one, &WF_(1, iw), &i1);
        sgemv_("T", &im1, &m, &one, &AF_(1, i + 1), &lda, &AF_(1, i), &i1, &zero, &WF_(i + 1, iw), &i1);
        sgemv_("N", &im1, &m, &mone, &WF_(1, iw + 1), &ldw, &WF_(i + 1, iw), &i1, &one, &WF_(1, iw), &i1);
      }
      sscal_(&im1, &tau[i - 2], &WF_(1, iw), &i1);
      float alpha = -half * tau[i - 2] * dot_f(im1, &WF_(1, iw), &AF_(1, i));
      saxpy_(&im1, &alpha, &AF_(1, i), &i1, &WF_(1, iw), &i1);
    }
  }
}

static void sytd2_u_f(lint n, float *a, lint lda, float *d, float *e, float *tau) {
  const float one = 1, mone = -1, half = 0.5f; const lint i1 = 1;
  for (lint i = n - 1; i >= 1; i--) {
    float taui;
    slarfg_(&i, &AF_(i, i + 1), &AF_(1, i + 1), &i1, &taui);
    e[i - 1] = AF_(i, i + 1);
    if (taui != 0) {
      AF_(i, i + 1) = one;
      symv_f(i, taui, a, lda, &AF_(1, i + 1), tau);
      float alpha = -half * taui * dot_f(i, tau, &AF_(1, i + 1));
      saxpy_(&i, &alpha, &AF_(1, i + 1), &i1, tau, &i1);
      ssyr2_("U", &i, &mone, &AF_(1, i + 1), &i1, tau, &i1, a, &lda);
      AF_(i, i + 1) = e[i - 1];
    }
    d[i] = AF_(i + 1, i + 1);
    tau[i - 1] = taui;
  }
  d[0] = AF_(1, 1);
}

static void sytrd_u_f(lint n, float *a, lint lda, float *d, float *e, float *tau, float *work) {
  const lint nb = 32, nx = 32; const float mone = -1, one = 1;
  lint kk = 0;
  if (n > nx) {
    kk = n - ((n - nx + nb - 1) / nb) * nb;
    for (lint i = n - nb + 1; i >= kk + 1; i -= nb) {
      lint m = i + nb - 1, im1 = i - 1;
      latrd_u_f(m, nb, a, lda, e, tau, work, n);
      ssyr2k_("U", "N", &im1, &nb, &mone, &AF_(1, i), &lda, work, &n, &one, a, &lda);
      for (lint j = i; j <= i + nb - 1; j++) { AF_(j - 1, j) = e[j - 2]; d[j - 1] = AF_(j, j); }
    }
  } else kk = n;
  sytd2_u_f(kk, a, lda, d, e, tau);
}

/* Fortran: call xtb_ssytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32 */
void xtb_ssytrd_upper(const lint *n, float *a, const lint *lda, float *d, float *e, float *tau, float *work) {
  sytrd_u_f(*n, a, *lda, d, e, tau, work);
}

/* Back-transformation Z = Q Z with Q from xtb_dsytrd_upper / xtb_ssytrd_upper: what
 * xormtr('L', 'U', 'N', n, m, a, lda, tau, z, ldz) computes, in xormql's blocks of nb = 32
 * reflectors (block i .. i + ib - 1 acts on rows 1 .. i + ib - 1, blocks in increasing i),
 * each applied by three gemm calls instead of xlarfb: V^T stored explicitly (W = V^T Z as
 * gemm 'N', 'N'), T from xlarft('B', 'C') made dense, W = T W, Z -= V W. Accelerate's
 * larfb runs well below its gemm, and gemm 'T', 'N' for V^T Z below 'N', 'N' on a
 * transposed copy: 1.2-1.4x faster than dormtr at n = 412-776, m = n / 2, one process,
 * and 1.1-1.25x with 8 processes x 1 thread (scripts/kernels/ormtr_gemm.c). Same result up
 * to rounding. work of size 32 * (2 * n + 2 * m + 32). */
extern void dlarft_(const char *, const char *, const lint *, const lint *, const double *, const lint *,
                    const double *, double *, const lint *);
extern void slarft_(const char *, const char *, const lint *, const lint *, const float *, const lint *,
                    const float *, float *, const lint *);
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *,
                   const float *, const lint *, const float *, const lint *, const float *, float *, const lint *);

#define ORMTR_UPPER(NAME, T, LARFT, GEMM)                                                         \
void NAME(const lint *np, const lint *mp, const T *a, const lint *ldap, const T *tau, T *z,       \
          const lint *ldzp, T *work) {                                                            \
  const lint n = *np, m = *mp, lda = *ldap, ldz = *ldzp, nb = 32, k = n - 1;                      \
  const T one = 1, zero = 0, mone = -1;                                                           \
  T *v = work, *vt = v + (size_t)n * nb, *t = vt + (size_t)n * nb, *w = t + nb * nb,              \
    *w2 = w + (size_t)nb * m;                                                                     \
  if (m <= 0) return;                                                                             \
  for (lint i = 1; i <= k; i += nb) {                                                             \
    lint ib = k - i + 1 < nb ? k - i + 1 : nb, mi = i + ib - 1;                                   \
    /* Reflector r = i + c: v(1 .. r - 1) = a(1 .. r - 1, r + 1), v(r) = 1, zero below. */        \
    for (lint c = 0; c < ib; c++) {                                                               \
      const T *src = a + (size_t)(i + c) * lda;                                                   \
      T *dst = v + (size_t)c * mi;                                                                \
      lint r = i + c;                                                                             \
      for (lint q = 0; q < r - 1; q++) dst[q] = src[q];                                           \
      dst[r - 1] = 1;                                                                             \
      for (lint q = r; q < mi; q++) dst[q] = 0;                                                   \
    }                                                                                             \
    for (lint c = 0; c < ib; c++)                                                                 \
      for (lint q = 0; q < mi; q++) vt[c + q * ib] = v[q + (size_t)c * mi];                       \
    LARFT("B", "C", &mi, &ib, v, &mi, tau + i - 1, t, &ib);                                       \
    for (lint c = 1; c < ib; c++)                                                                 \
      for (lint q = 0; q < c; q++) t[q + c * ib] = 0;                                             \
    GEMM("N", "N", &ib, &m, &mi, &one, vt, &ib, z, &ldz, &zero, w, &ib);                          \
    GEMM("N", "N", &ib, &m, &ib, &one, t, &ib, w, &ib, &zero, w2, &ib);                           \
    GEMM("N", "N", &mi, &m, &ib, &mone, v, &mi, w2, &ib, &one, z, &ldz);                          \
  }                                                                                               \
}

/* Fortran: call xtb_dormtr_upper(n, m, a, lda, tau, z, ldz, work) */
ORMTR_UPPER(xtb_dormtr_upper, double, dlarft_, dgemm_)
/* Fortran: call xtb_sormtr_upper(n, m, a, lda, tau, z, ldz, work) */
ORMTR_UPPER(xtb_sormtr_upper, float, slarft_, sgemm_)
