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
 * once for four columns at a time (~50 GB/s on M1 Max against ~30 GB/s for
 * Accelerate's dsymv; dsytrd 30% faster at n = 412-776). Elsewhere it calls dsymv. */

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

#if defined(__aarch64__)
// y = alpha * A x, A symmetric, upper triangle, n x n, column-major with lda
static void symv_upper(int n, double alpha, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0.0;
  int j0 = 0;
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

#define A_(i, j) a[((i) - 1) + (size_t)((j) - 1) * lda]
#define W_(i, j) w[((i) - 1) + (size_t)((j) - 1) * ldw]

// LAPACK dlatrd, UPLO = 'U'
static void latrd_u(lint n, lint nb, double *a, lint lda, double *e, double *tau, double *w, lint ldw) {
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
      symv(im1, one, a, lda, &A_(1, i), &W_(1, iw));
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
static void sytd2_u(lint n, double *a, lint lda, double *d, double *e, double *tau) {
  const double one = 1, mone = -1, half = 0.5; const lint i1 = 1;
  for (lint i = n - 1; i >= 1; i--) {
    double taui;
    dlarfg_(&i, &A_(i, i + 1), &A_(1, i + 1), &i1, &taui);
    e[i - 1] = A_(i, i + 1);
    if (taui != 0) {
      A_(i, i + 1) = one;
      symv(i, taui, a, lda, &A_(1, i + 1), tau);
      double alpha = -half * taui * ddot_(&i, tau, &i1, &A_(1, i + 1), &i1);
      daxpy_(&i, &alpha, &A_(1, i + 1), &i1, tau, &i1);
      dsyr2_("U", &i, &mone, &A_(1, i + 1), &i1, tau, &i1, a, &lda);
      A_(i, i + 1) = e[i - 1];
    }
    d[i] = A_(i + 1, i + 1);
    tau[i - 1] = taui;
  }
  d[0] = A_(1, 1);
}

// LAPACK dsytrd, UPLO = 'U', nb = 32, crossover nx = 32 (ilaenv defaults); work >= n * nb
static void sytrd_u(lint n, double *a, lint lda, double *d, double *e, double *tau, double *work) {
  const lint nb = 32, nx = 32; const double mone = -1, one = 1;
  lint kk = 0;
  if (n > nx) {
    kk = n - ((n - nx + nb - 1) / nb) * nb;
    for (lint i = n - nb + 1; i >= kk + 1; i -= nb) {
      lint m = i + nb - 1, im1 = i - 1;
      latrd_u(m, nb, a, lda, e, tau, work, n);
      dsyr2k_("U", "N", &im1, &nb, &mone, &A_(1, i), &lda, work, &n, &one, a, &lda);
      for (lint j = i; j <= i + nb - 1; j++) { A_(j - 1, j) = e[j - 2]; d[j - 1] = A_(j, j); }
    }
  } else kk = n;
  sytd2_u(kk, a, lda, d, e, tau);
}


/* Fortran: call xtb_dsytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32 */
void xtb_dsytrd_upper(const lint *n, double *a, const lint *lda, double *d, double *e, double *tau, double *work) {
  sytrd_u(*n, a, *lda, d, e, tau, work);
}
