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
 * expects. The changes, on aarch64: the symmetric matrix-vector product, the BLAS2
 * core of the reduction, is a hand-vectorized kernel that reads the triangle once for
 * eight columns at a time (~55-65 GB/s on M1 Max against ~30 GB/s for Accelerate's
 * dsymv; dsytrd 30% faster at n = 412-776), and dlatrd's six panel gemv calls per
 * column are three fused passes (panel_update, panel_dots). Elsewhere it calls dsymv
 * and dgemv.
 *
 * xtb_dsytrd_full is the same reduction on the full matrix (both triangles stored and
 * updated): dgemv 'T' (A^T x = A x) instead of the symmetric product, two dgemm instead of
 * dsyr2k, two dger instead of dsyr2. Twice the arithmetic, but Accelerate's full-matrix
 * kernels are fast enough that with a leading dimension padded to 64 bytes it is 2-19%
 * faster at n = 412-776 (dgemv 'T' reads the matrix at 100-155 GB/s, the symmetric product
 * at ~50). xtb_dsymmetrize_pad builds its input from the upper triangle. */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#define XTB_OMP_PRAGMA(x) _Pragma(#x)
/* The OpenMP threads the calling thread would get (its own ICV, which omp_set_num_threads
 * sets), or 1 inside a parallel region. */
static int xtb_omp_threads(void) { return omp_in_parallel() ? 1 : omp_get_max_threads(); }
#else
#define XTB_OMP_PRAGMA(x)
static int xtb_omp_threads(void) { return 1; }
#endif
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
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *,
                   const float *, const lint *, const float *, const lint *, const float *, float *, const lint *);
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
// y(0 .. n - 1) += the contributions of columns jbeg .. n - 1 of the upper triangle (jbeg a multiple of 8)
static void symv_cols(int jbeg, int n, const double *A, int lda, const double *x, double *y) {
  int j0 = jbeg;
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
}

static void symv_upper(int n, double alpha, const double *A, int lda, const double *x, double *y) {
  for (int i = 0; i < n; i++) y[i] = 0;
  symv_cols(0, n, A, lda, x, y);
  if (alpha != 1.0) for (int i = 0; i < n; i++) y[i] *= alpha;
}
#endif

#if defined(__aarch64__)
/* dlatrd's panel products, fused (each pair of LAPACK's gemv calls reads the panels of A and W
 * that share their rows): panel_update y -= P1 c1 + P2 c2 in one pass over y, eight columns
 * per load of y; panel_dots u1 = P1^T x, u2 = P2^T x in one pass over x, eight dot products
 * per load of x. P1 and P2 are rows x m, column-major; c1, c2, u1, u2 contiguous. */
#define PANEL_KERNELS(T, V, W, S, SFX)                                                            \
static void panel_update##SFX(lint rows, lint m, const T *p1, lint ld1, const T *c1, const T *p2,  \
                              lint ld2, const T *c2, T *y) {                                     \
  lint c = 0;                                                                                     \
  for (; c + 4 <= m; c += 4) {                                                                    \
    const T *a0 = p1 + (size_t)c * ld1, *a1 = a0 + ld1, *a2 = a1 + ld1, *a3 = a2 + ld1;           \
    const T *b0 = p2 + (size_t)c * ld2, *b1 = b0 + ld2, *b2 = b1 + ld2, *b3 = b2 + ld2;           \
    V ka0 = vdupq_n_##S(c1[c]), ka1 = vdupq_n_##S(c1[c + 1]), ka2 = vdupq_n_##S(c1[c + 2]),        \
      ka3 = vdupq_n_##S(c1[c + 3]), kb0 = vdupq_n_##S(c2[c]), kb1 = vdupq_n_##S(c2[c + 1]),        \
      kb2 = vdupq_n_##S(c2[c + 2]), kb3 = vdupq_n_##S(c2[c + 3]);                                 \
    lint i = 0;                                                                                   \
    for (; i + W <= rows; i += W) {                                                               \
      V yi = vld1q_##S(y + i);                                                                    \
      yi = vfmsq_##S(yi, vld1q_##S(a0 + i), ka0); yi = vfmsq_##S(yi, vld1q_##S(b0 + i), kb0);     \
      yi = vfmsq_##S(yi, vld1q_##S(a1 + i), ka1); yi = vfmsq_##S(yi, vld1q_##S(b1 + i), kb1);     \
      yi = vfmsq_##S(yi, vld1q_##S(a2 + i), ka2); yi = vfmsq_##S(yi, vld1q_##S(b2 + i), kb2);     \
      yi = vfmsq_##S(yi, vld1q_##S(a3 + i), ka3); yi = vfmsq_##S(yi, vld1q_##S(b3 + i), kb3);     \
      vst1q_##S(y + i, yi);                                                                       \
    }                                                                                             \
    for (; i < rows; i++)                                                                         \
      y[i] -= a0[i] * c1[c] + b0[i] * c2[c] + a1[i] * c1[c + 1] + b1[i] * c2[c + 1]               \
            + a2[i] * c1[c + 2] + b2[i] * c2[c + 2] + a3[i] * c1[c + 3] + b3[i] * c2[c + 3];      \
  }                                                                                               \
  for (; c < m; c++) {                                                                            \
    const T *a0 = p1 + (size_t)c * ld1, *b0 = p2 + (size_t)c * ld2;                               \
    V ka = vdupq_n_##S(c1[c]), kb = vdupq_n_##S(c2[c]);                                           \
    lint i = 0;                                                                                   \
    for (; i + W <= rows; i += W) {                                                               \
      V yi = vld1q_##S(y + i);                                                                    \
      yi = vfmsq_##S(yi, vld1q_##S(a0 + i), ka); yi = vfmsq_##S(yi, vld1q_##S(b0 + i), kb);       \
      vst1q_##S(y + i, yi);                                                                       \
    }                                                                                             \
    for (; i < rows; i++) y[i] -= a0[i] * c1[c] + b0[i] * c2[c];                                  \
  }                                                                                               \
}                                                                                                 \
static void panel_dots##SFX(lint rows, lint m, const T *p1, lint ld1, const T *p2, lint ld2,       \
                            const T *x, T *u1, T *u2) {                                           \
  lint c = 0;                                                                                     \
  for (; c + 4 <= m; c += 4) {                                                                    \
    const T *a0 = p1 + (size_t)c * ld1, *a1 = a0 + ld1, *a2 = a1 + ld1, *a3 = a2 + ld1;           \
    const T *b0 = p2 + (size_t)c * ld2, *b1 = b0 + ld2, *b2 = b1 + ld2, *b3 = b2 + ld2;           \
    V sa0 = vdupq_n_##S(0), sa1 = sa0, sa2 = sa0, sa3 = sa0, sb0 = sa0, sb1 = sa0, sb2 = sa0,       \
      sb3 = sa0;                                                                                  \
    lint i = 0;                                                                                   \
    for (; i + W <= rows; i += W) {                                                               \
      V xi = vld1q_##S(x + i);                                                                    \
      sa0 = vfmaq_##S(sa0, vld1q_##S(a0 + i), xi); sb0 = vfmaq_##S(sb0, vld1q_##S(b0 + i), xi);   \
      sa1 = vfmaq_##S(sa1, vld1q_##S(a1 + i), xi); sb1 = vfmaq_##S(sb1, vld1q_##S(b1 + i), xi);   \
      sa2 = vfmaq_##S(sa2, vld1q_##S(a2 + i), xi); sb2 = vfmaq_##S(sb2, vld1q_##S(b2 + i), xi);   \
      sa3 = vfmaq_##S(sa3, vld1q_##S(a3 + i), xi); sb3 = vfmaq_##S(sb3, vld1q_##S(b3 + i), xi);   \
    }                                                                                             \
    T ta0 = vaddvq_##S(sa0), ta1 = vaddvq_##S(sa1), ta2 = vaddvq_##S(sa2), ta3 = vaddvq_##S(sa3);  \
    T tb0 = vaddvq_##S(sb0), tb1 = vaddvq_##S(sb1), tb2 = vaddvq_##S(sb2), tb3 = vaddvq_##S(sb3);  \
    for (; i < rows; i++) {                                                                       \
      ta0 += a0[i] * x[i]; ta1 += a1[i] * x[i]; ta2 += a2[i] * x[i]; ta3 += a3[i] * x[i];         \
      tb0 += b0[i] * x[i]; tb1 += b1[i] * x[i]; tb2 += b2[i] * x[i]; tb3 += b3[i] * x[i];         \
    }                                                                                             \
    u1[c] = ta0; u1[c + 1] = ta1; u1[c + 2] = ta2; u1[c + 3] = ta3;                               \
    u2[c] = tb0; u2[c + 1] = tb1; u2[c + 2] = tb2; u2[c + 3] = tb3;                               \
  }                                                                                               \
  for (; c < m; c++) {                                                                            \
    const T *a0 = p1 + (size_t)c * ld1, *b0 = p2 + (size_t)c * ld2;                               \
    V sa = vdupq_n_##S(0), sb = sa;                                                               \
    lint i = 0;                                                                                   \
    for (; i + W <= rows; i += W) {                                                               \
      V xi = vld1q_##S(x + i);                                                                    \
      sa = vfmaq_##S(sa, vld1q_##S(a0 + i), xi); sb = vfmaq_##S(sb, vld1q_##S(b0 + i), xi);       \
    }                                                                                             \
    T ta = vaddvq_##S(sa), tb = vaddvq_##S(sb);                                                   \
    for (; i < rows; i++) { ta += a0[i] * x[i]; tb += b0[i] * x[i]; }                             \
    u1[c] = ta; u2[c] = tb;                                                                       \
  }                                                                                               \
}
PANEL_KERNELS(double, float64x2_t, 2, f64, _d)
PANEL_KERNELS(float, float32x4_t, 4, f32, _f)
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
#if defined(__aarch64__)
  (void)mone; (void)zero;  // the fused panel kernels replace the gemv calls
#endif
  for (lint i = n; i >= n - nb + 1; i--) {
    lint iw = i - n + nb;
    if (i < n) {
      lint m = n - i;
#if defined(__aarch64__)
      double wr[32], ar[32];
      for (lint c = 0; c < m; c++) { wr[c] = W_(i, iw + 1 + c); ar[c] = A_(i, i + 1 + c); }
      panel_update_d(i, m, &A_(1, i + 1), lda, wr, &W_(1, iw + 1), ldw, ar, &A_(1, i));
#else
      dgemv_("N", &i, &m, &mone, &A_(1, i + 1), &lda, &W_(i, iw + 1), &ldw, &one, &A_(1, i), &i1);
      dgemv_("N", &i, &m, &mone, &W_(1, iw + 1), &ldw, &A_(i, i + 1), &lda, &one, &A_(1, i), &i1);
#endif
    }
    if (i > 1) {
      lint im1 = i - 1;
      dlarfg_(&im1, &A_(i - 1, i), &A_(1, i), &i1, &tau[i - 2]);
      e[i - 2] = A_(i - 1, i);
      A_(i - 1, i) = one;
      matvec(sym, im1, one, a, lda, &A_(1, i), &W_(1, iw));
      if (i < n) {
        lint m = n - i;
#if defined(__aarch64__)
        // u = W^T v and u2 = A^T v in one pass, then w -= A u + W u2 in one; W(i + 1 .. n, iw) ends as
        // u2, as with LAPACK's call order.
        double u[32];
        panel_dots_d(im1, m, &W_(1, iw + 1), ldw, &A_(1, i + 1), lda, &A_(1, i), u, &W_(i + 1, iw));
        panel_update_d(im1, m, &A_(1, i + 1), lda, u, &W_(1, iw + 1), ldw, &W_(i + 1, iw), &W_(1, iw));
#else
        dgemv_("T", &im1, &m, &one, &W_(1, iw + 1), &ldw, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &A_(1, i + 1), &lda, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
        dgemv_("T", &im1, &m, &one, &A_(1, i + 1), &lda, &A_(1, i), &i1, &zero, &W_(i + 1, iw), &i1);
        dgemv_("N", &im1, &m, &mone, &W_(1, iw + 1), &ldw, &W_(i + 1, iw), &i1, &one, &W_(1, iw), &i1);
#endif
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
// y(0 .. n - 1) += the contributions of columns jbeg .. n - 1 of the upper triangle (jbeg a multiple of 8)
static void symv_cols_f(int jbeg, int n, const float *A, int lda, const float *x, float *y) {
  int j0 = jbeg;
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
}

static void symv_upper_f(int n, float alpha, const float *A, int lda, const float *x, float *y) {
  for (int i = 0; i < n; i++) y[i] = 0;
  symv_cols_f(0, n, A, lda, x, y);
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
#if defined(__aarch64__)
  (void)mone; (void)zero;  // the fused panel kernels replace the gemv calls
#endif
  for (lint i = n; i >= n - nb + 1; i--) {
    lint iw = i - n + nb;
    if (i < n) {
      lint m = n - i;
#if defined(__aarch64__)
      float wr[32], ar[32];
      for (lint c = 0; c < m; c++) { wr[c] = WF_(i, iw + 1 + c); ar[c] = AF_(i, i + 1 + c); }
      panel_update_f(i, m, &AF_(1, i + 1), lda, wr, &WF_(1, iw + 1), ldw, ar, &AF_(1, i));
#else
      sgemv_("N", &i, &m, &mone, &AF_(1, i + 1), &lda, &WF_(i, iw + 1), &ldw, &one, &AF_(1, i), &i1);
      sgemv_("N", &i, &m, &mone, &WF_(1, iw + 1), &ldw, &AF_(i, i + 1), &lda, &one, &AF_(1, i), &i1);
#endif
    }
    if (i > 1) {
      lint im1 = i - 1;
      slarfg_(&im1, &AF_(i - 1, i), &AF_(1, i), &i1, &tau[i - 2]);
      e[i - 2] = AF_(i - 1, i);
      AF_(i - 1, i) = one;
      symv_f(im1, one, a, lda, &AF_(1, i), &WF_(1, iw));
      if (i < n) {
        lint m = n - i;
#if defined(__aarch64__)
        float u[32];
        panel_dots_f(im1, m, &WF_(1, iw + 1), ldw, &AF_(1, i + 1), lda, &AF_(1, i), u, &WF_(i + 1, iw));
        panel_update_f(im1, m, &AF_(1, i + 1), lda, u, &WF_(1, iw + 1), ldw, &WF_(i + 1, iw), &WF_(1, iw));
#else
        sgemv_("T", &im1, &m, &one, &WF_(1, iw + 1), &ldw, &AF_(1, i), &i1, &zero, &WF_(i + 1, iw), &i1);
        sgemv_("N", &im1, &m, &mone, &AF_(1, i + 1), &lda, &WF_(i + 1, iw), &i1, &one, &WF_(1, iw), &i1);
        sgemv_("T", &im1, &m, &one, &AF_(1, i + 1), &lda, &AF_(1, i), &i1, &zero, &WF_(i + 1, iw), &i1);
        sgemv_("N", &im1, &m, &mone, &WF_(1, iw + 1), &ldw, &WF_(i + 1, iw), &i1, &one, &WF_(1, iw), &i1);
#endif
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

/* The reduction on the caller's OpenMP threads: one team for the whole call (a parallel region
 * per column costs more than the column), synchronized by a spinning barrier (about 0.1 us
 * with 4 threads). Per column of dlatrd: rows of the panel update split across threads ->
 * barrier -> xlarfg on thread 0 -> barrier -> symv split by columns (equal triangle areas)
 * into per-thread vectors, and panel_dots split by rows -> barrier -> sum of the vectors,
 * panel_update, scaling and the partial dot, split by rows -> barrier -> axpy, same rows (and
 * the next column's panel update, same rows again). The syr2k after each panel is split by
 * columns (gemm above the diagonal blocks); the leading columns below order 128 run serially
 * on thread 0. Same result up to rounding, and the same bits for the same thread count; one
 * thread takes the serial path. */
#if defined(_OPENMP) && defined(__aarch64__)
#include <stdatomic.h>

typedef struct {
  _Alignas(128) atomic_int count;
  _Alignas(128) atomic_int generation;
} xtb_barrier;

static void xtb_barrier_wait(xtb_barrier *b, int threads) {
  int g = atomic_load_explicit(&b->generation, memory_order_acquire);
  if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) == threads - 1) {
    atomic_store_explicit(&b->count, 0, memory_order_relaxed);
    atomic_store_explicit(&b->generation, g + 1, memory_order_release);
  } else {
    while (atomic_load_explicit(&b->generation, memory_order_acquire) == g) __asm__ volatile("yield");
  }
}

// Start of part t of [0, len) in T parts, at multiples of 32 elements (whole cache lines)
static lint team_rows(lint len, int t, int threads) {
  lint r = ((len * t / threads + 31) / 32) * 32;
  return r < len ? r : len;
}

// Start of the columns of thread t of the upper triangle of order len: equal areas, multiples of 8
static lint team_cols(lint len, int t, int threads) {
  if (t == threads) return len;
  lint c = (lint)(len * sqrt((double)t / threads) / 8 + 0.5) * 8;
  return c < len ? c : len;
}

#define SYTRD_TEAM(SFX, T, FMA, LARFG, SYR2K, GEMM, SYMV_COLS, PANEL_UPDATE, PANEL_DOTS, REST)    \
static void latrd_team##SFX(int t, int threads, lint n, lint nb, T *a, lint lda, T *e, T *tau,    \
                            T *w, lint ldw, T *ybuf, lint ldy, T *ubuf, T *dbuf, T *shared,       \
                            xtb_barrier *bar) {                                                   \
  const lint i1 = 1;                                                                              \
  T wlast_alpha = 0;  /* the last column's W(i, iw + 1), recomputed by every thread */            \
  for (lint i = n; i >= n - nb + 1; i--) {                                                        \
    lint iw = i - n + nb;                                                                         \
    if (i < n) {                                                                                  \
      lint m = n - i, r0 = team_rows(i, t, threads), r1 = team_rows(i, t + 1, threads);           \
      T wr[32], ar[32];                                                                           \
      wr[0] = wlast_alpha; ar[0] = a[(i - 1) + (size_t)i * lda];                                  \
      for (lint c = 1; c < m; c++) {                                                              \
        wr[c] = w[(i - 1) + (size_t)(iw + c) * ldw]; ar[c] = a[(i - 1) + (size_t)(i + c) * lda];  \
      }                                                                                           \
      if (r1 > r0)                                                                                \
        PANEL_UPDATE(r1 - r0, m, a + r0 + (size_t)i * lda, lda, wr, w + r0 + (size_t)iw * ldw,    \
                     ldw, ar, a + r0 + (size_t)(i - 1) * lda);                                    \
    }                                                                                             \
    if (i == 1) break;                                                                            \
    lint im1 = i - 1, m = n - i;                                                                  \
    T *v = a + (size_t)(i - 1) * lda, *wc = w + (size_t)(iw - 1) * ldw;                           \
    xtb_barrier_wait(bar, threads);                                                               \
    if (t == 0) {                                                                                 \
      LARFG(&im1, &v[i - 2], v, &i1, &tau[i - 2]);                                                \
      e[i - 2] = v[i - 2];                                                                        \
      v[i - 2] = 1;                                                                               \
    }                                                                                             \
    xtb_barrier_wait(bar, threads);                                                               \
    /* symv: columns c0 .. c1 - 1 into ybuf_t(0 .. c1 - 1) */                                     \
    lint cend[65];                                                                                \
    for (int q = 0; q <= threads; q++) cend[q] = team_cols(im1, q, threads);                      \
    lint c0 = cend[t], c1 = cend[t + 1];                                                          \
    T *y = ybuf + (size_t)t * ldy;                                                                \
    for (lint r = 0; r < c1; r++) y[r] = 0;                                                       \
    if (c1 > c0) SYMV_COLS(c0, c1, a, lda, v, y);                                                 \
    lint r0 = team_rows(im1, t, threads), r1 = team_rows(im1, t + 1, threads);                    \
    T *u = ubuf + (size_t)t * 64;                                                                 \
    for (lint c = 0; c < m; c++) { u[c] = 0; u[32 + c] = 0; }                                     \
    if (m > 0 && r1 > r0)                                                                         \
      PANEL_DOTS(r1 - r0, m, w + r0 + (size_t)iw * ldw, ldw, a + r0 + (size_t)i * lda, lda,       \
                 v + r0, u, u + 32);                                                              \
    xtb_barrier_wait(bar, threads);                                                               \
    T us[64];                                                                                     \
    for (lint c = 0; c < m; c++) {                                                                \
      T s1 = 0, s2 = 0;                                                                           \
      for (int s = 0; s < threads; s++) {                                                         \
        s1 += ubuf[(size_t)s * 64 + c]; s2 += ubuf[(size_t)s * 64 + 32 + c];                      \
      }                                                                                           \
      us[c] = s1; us[32 + c] = s2;                                                                \
    }                                                                                             \
    if (t == 0) for (lint c = 0; c < m; c++) w[i + c + (size_t)(iw - 1) * ldw] = us[32 + c];      \
    for (lint r = r0; r < r1; r++) {                                                              \
      T s = 0;                                                                                    \
      for (int q = 0; q < threads; q++) if (r < cend[q + 1]) s += ybuf[(size_t)q * ldy + r];      \
      wc[r] = s;                                                                                  \
    }                                                                                             \
    if (m > 0 && r1 > r0)                                                                         \
      PANEL_UPDATE(r1 - r0, m, a + r0 + (size_t)i * lda, lda, us, w + r0 + (size_t)iw * ldw, ldw, \
                   us + 32, wc + r0);                                                             \
    T taui = tau[i - 2], dot = 0;                                                                 \
    for (lint r = r0; r < r1; r++) { wc[r] *= taui; dot += wc[r] * v[r]; }                        \
    dbuf[(size_t)t * 32] = dot;                                                                   \
    if (r0 < im1 && im1 <= r1) shared[0] = wc[im1 - 1];                                           \
    xtb_barrier_wait(bar, threads);                                                               \
    T sum = 0;                                                                                    \
    for (int s = 0; s < threads; s++) sum += dbuf[(size_t)s * 32];                                \
    T alpha = -(T)0.5 * taui * sum;                                                               \
    for (lint r = r0; r < r1; r++) wc[r] = FMA(alpha, v[r], wc[r]);                               \
    wlast_alpha = FMA(alpha, v[im1 - 1], shared[0]);                                              \
  }                                                                                               \
}                                                                                                 \
static void sytrd_team##SFX(lint n, T *a, lint lda, T *d, T *e, T *tau, T *work, int threads) {   \
  const lint nb = 32, nx = 32, tail = 128; const T mone = -1, one = 1;                            \
  lint kk = n - ((n - nx + nb - 1) / nb) * nb, ldy = (n + 31) / 32 * 32;                          \
  T *ybuf = malloc(sizeof(T) * ((size_t)threads * (ldy + 64 + 32) + 32));                         \
  T *ubuf = ybuf + (size_t)threads * ldy, *dbuf = ubuf + (size_t)threads * 64,                    \
    *shared = dbuf + (size_t)threads * 32;                                                        \
  xtb_barrier bar; atomic_init(&bar.count, 0); atomic_init(&bar.generation, 0);                   \
  lint last = n - nb + 1;  /* first column of the last panel the team reduces */                  \
  while (last - nb >= kk + 1 && last - 1 >= tail) last -= nb;                                     \
  XTB_OMP_PRAGMA(omp parallel num_threads(threads))                                               \
  {                                                                                               \
    int t = omp_get_thread_num(), team = omp_get_num_threads();                                   \
    for (lint i = n - nb + 1; i >= last; i -= nb) {                                               \
      lint m = i + nb - 1, im1 = i - 1;                                                           \
      latrd_team##SFX(t, team, m, nb, a, lda, e, tau, work, n, ybuf, ldy, ubuf, dbuf, shared,     \
                      &bar);                                                                      \
      xtb_barrier_wait(&bar, team);                                                               \
      {  /* columns c0 .. c1 - 1 of the update, equal areas */                                    \
        lint c0 = team_cols(im1, t, team), c1 = team_cols(im1, t + 1, team), nc = c1 - c0;        \
        const T *v = &a[(size_t)(i - 1) * lda];                                                   \
        if (c0 > 0 && nc > 0) {                                                                   \
          T *ac = &a[(size_t)c0 * lda];                                                           \
          GEMM("N", "T", &c0, &nc, &nb, &mone, v, &lda, work + c0, &n, &one, ac, &lda);           \
          GEMM("N", "T", &c0, &nc, &nb, &mone, work, &n, v + c0, &lda, &one, ac, &lda);           \
        }                                                                                         \
        if (nc > 0)                                                                               \
          SYR2K("U", "N", &nc, &nb, &mone, v + c0, &lda, work + c0, &n, &one,                     \
                &a[c0 + (size_t)c0 * lda], &lda);                                                 \
      }                                                                                           \
      xtb_barrier_wait(&bar, team);                                                               \
      if (t == 0) {                                                                               \
        for (lint j = i; j <= i + nb - 1; j++) {                                                  \
          a[(j - 2) + (size_t)(j - 1) * lda] = e[j - 2];                                          \
          d[j - 1] = a[(j - 1) + (size_t)(j - 1) * lda];                                          \
        }                                                                                         \
      }                                                                                           \
      xtb_barrier_wait(&bar, team);                                                               \
    }                                                                                             \
  }                                                                                               \
  free(ybuf);                                                                                     \
  /* the leading last - 1 columns, serially */                                                    \
  REST(last - 1, a, lda, d, e, tau, work);                                                        \
}
static void sytrd_rest_d(lint n, double *a, lint lda, double *d, double *e, double *tau, double *work) {
  sytrd_u(1, n, a, lda, d, e, tau, work);
}
SYTRD_TEAM(_d, double, fma, dlarfg_, dsyr2k_, dgemm_, symv_cols, panel_update_d, panel_dots_d, sytrd_rest_d)
SYTRD_TEAM(_f, float, fmaf, slarfg_, ssyr2k_, sgemm_, symv_cols_f, panel_update_f, panel_dots_f, sytrd_u_f)
#endif

/* The reduction on the caller's OpenMP threads from this order up, on at most one cluster of
 * performance cores (on the M1 Max, 4 cores per cluster, 6 and 8 threads were slower than 4).
 * Cores per cluster: hw.perflevel0.physicalcpu over its clusters, ceil(physicalcpu / cpusperl2),
 * so a cluster with disabled cores counts the enabled ones whether cpusperl2 reports enabled or
 * designed cores (6 performance cores, cpusperl2 4 or 3: 3); 4 where the sysctls are missing. */
enum { xtb_sytrd_team_min = 256 };
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
static int xtb_sytrd_team_max(void) {
  static int cached = 0;
  if (cached == 0) {
    int cores = 0, shared = 0;
#if defined(__APPLE__)
    size_t size = sizeof(int);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, NULL, 0) != 0) cores = 0;
    size = sizeof(int);
    if (sysctlbyname("hw.perflevel0.cpusperl2", &shared, &size, NULL, 0) != 0) shared = 0;
#endif
    cached = cores > 0 && shared > 0 ? cores / ((cores + shared - 1) / shared) : 4;
  }
  return cached;
}

/* Fortran: call xtb_dsytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32 */
void xtb_dsytrd_upper(const lint *n, double *a, const lint *lda, double *d, double *e, double *tau, double *work) {
#if defined(_OPENMP) && defined(__aarch64__)
  int threads = xtb_omp_threads(), most = xtb_sytrd_team_max();
  if (threads > 1 && *n >= xtb_sytrd_team_min) {
    sytrd_team_d(*n, a, *lda, d, e, tau, work, threads > most ? most : threads);
    return;
  }
#endif
  sytrd_u(1, *n, a, *lda, d, e, tau, work);
}

/* Fortran: call xtb_ssytrd_upper(n, a, lda, d, e, tau, work), work of size n * 32 */
void xtb_ssytrd_upper(const lint *n, float *a, const lint *lda, float *d, float *e, float *tau, float *work) {
#if defined(_OPENMP) && defined(__aarch64__)
  int threads = xtb_omp_threads(), most = xtb_sytrd_team_max();
  if (threads > 1 && *n >= xtb_sytrd_team_min) {
    sytrd_team_f(*n, a, *lda, d, e, tau, work, threads > most ? most : threads);
    return;
  }
#endif
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
static void NAME##_serial(const lint *np, const lint *mp, const T *a, const lint *ldap,           \
                          const T *tau, T *z, const lint *ldzp, T *work) {                        \
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

ORMTR_UPPER(xtb_dormtr_upper, double, dlarft_, dgemm_)
ORMTR_UPPER(xtb_sormtr_upper, float, slarft_, sgemm_)

/* With more than one OpenMP thread available to the caller (and not already inside a parallel
 * region), the columns of z are split into two blocks on two threads, each with its own
 * workspace; the blocks are independent. Two, not more: the blocks' gemm calls share the AMX
 * unit of a core cluster, and 4 or 8 threads were mostly slower than 2 (scripts/kernels/
 * stedc_omp.c, M1 Max, m = n / 2: 1.25-1.9x with 2 at n = 412-1788). Otherwise the serial
 * routine with the caller's work. */
#define ORMTR_UPPER_ENTRY(NAME, T)                                                                \
void NAME(const lint *np, const lint *mp, const T *a, const lint *ldap, const T *tau, T *z,       \
          const lint *ldzp, T *work) {                                                            \
  int threads = xtb_omp_threads() > 1 ? 2 : 1;                                                    \
  lint m = *mp;                                                                                   \
  if (threads <= 1 || m < 2 * threads) { NAME##_serial(np, mp, a, ldap, tau, z, ldzp, work); return; } \
  lint width = (m + threads - 1) / threads, blocks = (m + width - 1) / width;                     \
  XTB_OMP_PRAGMA(omp parallel for num_threads(threads) schedule(static, 1))                       \
  for (lint blk = 0; blk < blocks; blk++) {                                                       \
    lint c0 = blk * width, cols = m - c0 < width ? m - c0 : width;                                \
    T *w = malloc(sizeof(T) * 32 * (2 * (size_t)*np + 2 * (size_t)cols + 32));                   \
    NAME##_serial(np, &cols, a, ldap, tau, z + (size_t)c0 * *ldzp, ldzp, w);                      \
    free(w);                                                                                      \
  }                                                                                               \
}

/* Fortran: call xtb_dormtr_upper(n, m, a, lda, tau, z, ldz, work) */
ORMTR_UPPER_ENTRY(xtb_dormtr_upper, double)
/* Fortran: call xtb_sormtr_upper(n, m, a, lda, tau, z, ldz, work) */
ORMTR_UPPER_ENTRY(xtb_sormtr_upper, float)

/* Divide and conquer for the symmetric tridiagonal eigenproblem (all eigenvectors, as
 * xstedc('I')) on the caller's OpenMP threads. With one thread (or inside a parallel region,
 * or n < 256) this is xstedc itself, bit for bit. Otherwise the top levels of the recursion run
 * here as OpenMP tasks: each level tears the matrix at n1 = n / 2 (a rank-one change, as
 * LAPACK's dlaed0), solves both halves at once, and merges them as dlaed1 does: deflation by
 * xlaed2, then dlaed3's steps with their independent parts as task loops (one xlaed4 call per
 * root, the stabilized eigenvectors of the rank-one problem per column, the gemm that forms
 * the merged eigenvectors per block of columns). The leaves are xstedc. Calls that can run
 * concurrently use Accelerate's new LAPACK interface ($NEWLAPACK), documented as thread-safe.
 * Q2 is n^2: xlaed2 stores the deflated vectors there with leading dimension n, beyond the
 * n1^2 + n2^2 its documentation gives (an overrun found by AddressSanitizer at n = 1788).
 * Same arguments as xstedc with COMPZ = 'I' (work and iwork are used only by the serial path).
 * The Swift implementation this ports: Sources/Diagonalization/Stages/ParallelDivideAndConquer.swift
 * in the parent repository (eigensolve alone, M1 Max, 8 threads: dstedc 2.3-2.8x faster at
 * n = 412-1788). */
#if defined(__APPLE__)
#define XTB_TS(name) __asm__("_" #name "$NEWLAPACK")
#else
#define XTB_TS(name) __asm__(#name "_")
#endif
extern void dstedc_ts(const char *, const lint *, double *, double *, double *, const lint *, double *,
                      const lint *, lint *, const lint *, lint *) XTB_TS(dstedc);
extern void sstedc_ts(const char *, const lint *, float *, float *, float *, const lint *, float *,
                      const lint *, lint *, const lint *, lint *) XTB_TS(sstedc);
extern void dlaed2_ts(lint *, const lint *, const lint *, double *, double *, const lint *, lint *, double *,
                      double *, double *, double *, double *, lint *, lint *, lint *, lint *, lint *) XTB_TS(dlaed2);
extern void slaed2_ts(lint *, const lint *, const lint *, float *, float *, const lint *, lint *, float *,
                      float *, float *, float *, float *, lint *, lint *, lint *, lint *, lint *) XTB_TS(slaed2);
extern void dlaed4_ts(const lint *, const lint *, const double *, const double *, double *, const double *,
                      double *, lint *) XTB_TS(dlaed4);
extern void slaed4_ts(const lint *, const lint *, const float *, const float *, float *, const float *,
                      float *, lint *) XTB_TS(slaed4);
extern void dlamrg_ts(const lint *, const lint *, const double *, const lint *, const lint *, lint *) XTB_TS(dlamrg);
extern void slamrg_ts(const lint *, const lint *, const float *, const lint *, const lint *, lint *) XTB_TS(slamrg);
extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void sstedc_(const char *, const lint *, float *, float *, float *, const lint *, float *, const lint *,
                    lint *, const lint *, lint *);

#define STEDC_PARALLEL(P, T, GEMM)                                                                \
static lint P##dc_leaf(lint n, T *d, T *e, T *q, lint ldq) {                                      \
  lint lwork = 1 + 4 * n + n * n, liwork = 3 + 5 * n, info = 0;                                   \
  T *work = malloc(sizeof(T) * (size_t)lwork);                                                    \
  lint *iwork = malloc(sizeof(lint) * (size_t)liwork);                                            \
  P##stedc_ts("I", &n, d, e, q, &ldq, work, &lwork, iwork, &liwork, &info);                       \
  free(work); free(iwork);                                                                        \
  return info;                                                                                    \
}                                                                                                 \
/* dlaed3: the k roots (into d), the rank-one problem's eigenvectors, q(:, 0:k) = Q2 S. */        \
static lint P##dc_secular(lint k, lint n, lint n1, T *d, T *q, lint ldq, T rho, T *dlamda,       \
                          const T *q2, const lint *indx, const lint *ctot, T *w) {                \
  lint bad = 0;                                                                                   \
  XTB_OMP_PRAGMA(omp taskloop grainsize(8) shared(bad))                                           \
  for (lint j = 0; j < k; j++) {                                                                  \
    lint jj = j + 1, info = 0;                                                                    \
    P##laed4_ts(&k, &jj, dlamda, w, q + (size_t)j * ldq, &rho, d + j, &info);                     \
    if (info) { XTB_OMP_PRAGMA(omp atomic write) bad = info; }                                    \
  }                                                                                               \
  if (bad) return bad;                                                                            \
  if (k == 2) {                                                                                   \
    for (lint j = 0; j < 2; j++) {                                                                \
      T pair[2] = {q[(size_t)j * ldq], q[1 + (size_t)j * ldq]};                                   \
      q[(size_t)j * ldq] = pair[indx[0] - 1]; q[1 + (size_t)j * ldq] = pair[indx[1] - 1];         \
    }                                                                                             \
  } else if (k > 2) {                                                                             \
    T *sg = malloc(sizeof(T) * 2 * (size_t)k), *wn = sg + k;                                      \
    memcpy(sg, w, sizeof(T) * (size_t)k);                                                         \
    /* z recomputed from the roots, each i on its own, in dlaed3's product order. */              \
    XTB_OMP_PRAGMA(omp taskloop grainsize(16))                                                    \
    for (lint i = 0; i < k; i++) {                                                                \
      T prod = q[i + (size_t)i * ldq];                                                            \
      for (lint j = 0; j < k; j++)                                                                \
        if (j != i) prod *= q[i + (size_t)j * ldq] / (dlamda[i] - dlamda[j]);                     \
      T mag = sqrt(-prod);                                                                        \
      wn[i] = sg[i] < 0 ? -mag : mag;                                                             \
    }                                                                                             \
    XTB_OMP_PRAGMA(omp taskloop grainsize(16))                                                    \
    for (lint j = 0; j < k; j++) {                                                                \
      T *col = q + (size_t)j * ldq, *sv = malloc(sizeof(T) * (size_t)k), sum = 0;                 \
      for (lint i = 0; i < k; i++) { sv[i] = wn[i] / col[i]; sum += sv[i] * sv[i]; }              \
      T scale = 1 / sqrt(sum);                                                                    \
      for (lint i = 0; i < k; i++) col[i] = sv[indx[i] - 1] * scale;                              \
      free(sv);                                                                                   \
    }                                                                                             \
    free(sg);                                                                                     \
  }                                                                                               \
  lint n2 = n - n1, c0 = ctot[0], n12 = ctot[0] + ctot[1], n23 = ctot[1] + ctot[2];               \
  lint width = 64, blocks = (k + width - 1) / width;                                              \
  XTB_OMP_PRAGMA(omp taskloop grainsize(1))                                                       \
  for (lint blk = 0; blk < blocks; blk++) {                                                       \
    lint j0 = blk * width, cols = k - j0 < width ? k - j0 : width;                                \
    lint big = n12 > n23 ? n12 : n23;                                                             \
    T *sm = malloc(sizeof(T) * (size_t)(big > 0 ? big : 1) * cols), one = 1, zero = 0;            \
    for (lint j = 0; j < cols; j++)                                                               \
      memcpy(sm + (size_t)j * n23, q + c0 + (size_t)(j0 + j) * ldq, sizeof(T) * (size_t)n23);     \
    if (n23 > 0) GEMM("N", "N", &n2, &cols, &n23, &one, q2 + (size_t)n1 * n12, &n2, sm, &n23,     \
                      &zero, q + n1 + (size_t)j0 * ldq, &ldq);                                    \
    else for (lint j = 0; j < cols; j++) memset(q + n1 + (size_t)(j0 + j) * ldq, 0, sizeof(T) * (size_t)n2); \
    for (lint j = 0; j < cols; j++)                                                               \
      memcpy(sm + (size_t)j * n12, q + (size_t)(j0 + j) * ldq, sizeof(T) * (size_t)n12);          \
    if (n12 > 0) GEMM("N", "N", &n1, &cols, &n12, &one, q2, &n1, sm, &n12, &zero,                 \
                      q + (size_t)j0 * ldq, &ldq);                                                \
    else for (lint j = 0; j < cols; j++) memset(q + (size_t)(j0 + j) * ldq, 0, sizeof(T) * (size_t)n1); \
    free(sm);                                                                                     \
  }                                                                                               \
  return 0;                                                                                       \
}                                                                                                 \
/* dlaed1, then the result sorted ascending (as the next level up expects). */                    \
static lint P##dc_merge(lint n, lint n1, T *d, T *q, lint ldq, T rho) {                           \
  lint n2 = n - n1, k = 0, info = 0;                                                              \
  T *z = malloc(sizeof(T) * (3 * (size_t)n + (size_t)n * n)), *dlamda = z + n, *w = dlamda + n,  \
    *q2 = w + n;                                                                                  \
  lint *iw = malloc(sizeof(lint) * 5 * (size_t)n), *indxq = iw, *indx = iw + n, *indxc = indx + n, \
       *indxp = indxc + n, *coltyp = indxp + n;                                                   \
  for (lint i = 0; i < n1; i++) indxq[i] = i + 1;                                                 \
  for (lint i = 0; i < n2; i++) indxq[n1 + i] = i + 1;                                            \
  for (lint i = 0; i < n1; i++) z[i] = q[(n1 - 1) + (size_t)i * ldq];                             \
  for (lint i = 0; i < n2; i++) z[n1 + i] = q[n1 + (size_t)(n1 + i) * ldq];                       \
  P##laed2_ts(&k, &n, &n1, d, q, &ldq, indxq, &rho, z, dlamda, w, q2, indx, indxc, indxp, coltyp, \
              &info);                                                                             \
  if (!info && k > 0) {                                                                           \
    info = P##dc_secular(k, n, n1, d, q, ldq, rho, dlamda, q2, indxc, coltyp, w);                 \
    lint nk = n - k, up = 1, down = -1;                                                           \
    if (!info) P##lamrg_ts(&k, &nk, d, &up, &down, indxq);                                        \
  } else if (!info) {                                                                             \
    for (lint i = 0; i < n; i++) indxq[i] = i + 1;                                                \
  }                                                                                               \
  if (!info) {                                                                                    \
    T *sorted = malloc(sizeof(T) * ((size_t)n * n + n)), *ds = sorted + (size_t)n * n;            \
    for (lint i = 0; i < n; i++) ds[i] = d[indxq[i] - 1];                                         \
    XTB_OMP_PRAGMA(omp taskloop grainsize(32))                                                    \
    for (lint i = 0; i < n; i++)                                                                  \
      memcpy(sorted + (size_t)i * n, q + (size_t)(indxq[i] - 1) * ldq, sizeof(T) * (size_t)n);    \
    XTB_OMP_PRAGMA(omp taskloop grainsize(32))                                                    \
    for (lint i = 0; i < n; i++)                                                                  \
      memcpy(q + (size_t)i * ldq, sorted + (size_t)i * n, sizeof(T) * (size_t)n);                 \
    memcpy(d, ds, sizeof(T) * (size_t)n);                                                         \
    free(sorted);                                                                                 \
  }                                                                                               \
  free(z); free(iw);                                                                              \
  return info;                                                                                    \
}                                                                                                 \
static lint P##dc_level(lint n, T *d, T *e, T *q, lint ldq, int depth) {                          \
  if (depth <= 0 || n < 128) return P##dc_leaf(n, d, e, q, ldq);                                 \
  lint n1 = n / 2, n2 = n - n1, i1 = 0, i2 = 0;                                                   \
  T rho = e[n1 - 1];                                                                              \
  d[n1 - 1] -= fabs(rho);                                                                         \
  d[n1] -= fabs(rho);                                                                             \
  XTB_OMP_PRAGMA(omp task shared(i1))                                                             \
  i1 = P##dc_level(n1, d, e, q, ldq, depth - 1);                                                  \
  XTB_OMP_PRAGMA(omp task shared(i2))                                                             \
  i2 = P##dc_level(n2, d + n1, e + n1, q + n1 + (size_t)n1 * ldq, ldq, depth - 1);                \
  XTB_OMP_PRAGMA(omp taskwait)                                                                    \
  if (i1) return i1;                                                                              \
  if (i2) return i2 + n1;                                                                         \
  return P##dc_merge(n, n1, d, q, ldq, rho);                                                      \
}                                                                                                 \
void xtb_##P##stedc(const lint *np, T *d, T *e, T *z, const lint *ldzp, T *work, const lint *lwork, \
                    lint *iwork, const lint *liwork, lint *info) {                                \
  lint n = *np, ldz = *ldzp;                                                                      \
  int threads = xtb_omp_threads();                                                                \
  if (threads <= 1 || n < 256) {                                                                  \
    P##stedc_("I", np, d, e, z, ldzp, work, lwork, iwork, liwork, info);                          \
    return;                                                                                       \
  }                                                                                               \
  /* Scale to unit max norm, as xstedc does before dlaed0. */                                     \
  T norm = 0;                                                                                     \
  for (lint i = 0; i < n; i++) norm = fabs(d[i]) > norm ? fabs(d[i]) : norm;                      \
  for (lint i = 0; i < n - 1; i++) norm = fabs(e[i]) > norm ? fabs(e[i]) : norm;                  \
  for (lint j = 0; j < n; j++) memset(z + (size_t)j * ldz, 0, sizeof(T) * (size_t)n);            \
  *info = 0;                                                                                      \
  if (norm == 0) { for (lint i = 0; i < n; i++) z[i + (size_t)i * ldz] = 1; return; }             \
  for (lint i = 0; i < n; i++) d[i] /= norm;                                                      \
  for (lint i = 0; i < n - 1; i++) e[i] /= norm;                                                  \
  int depth = 0;                                                                                  \
  while ((1 << depth) < threads) depth++;                                                         \
  lint result = 0;                                                                                \
  XTB_OMP_PRAGMA(omp parallel num_threads(threads))                                               \
  XTB_OMP_PRAGMA(omp single)                                                                      \
  result = P##dc_level(n, d, e, z, ldz, depth);                                                   \
  *info = result;                                                                                 \
  for (lint i = 0; i < n; i++) d[i] *= norm;                                                      \
}
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *,
                   const float *, const lint *, const float *, const lint *, const float *, float *, const lint *);
/* Fortran: call xtb_dstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info), as dstedc('I', ...) */
STEDC_PARALLEL(d, double, dgemm_)
/* Fortran: call xtb_sstedc(n, d, e, z, ldz, work, lwork, iwork, liwork, info), as sstedc('I', ...) */
STEDC_PARALLEL(s, float, sgemm_)
