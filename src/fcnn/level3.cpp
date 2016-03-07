/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2012-2016
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

/** \file level3.cpp
 *  \brief Level 3 operations: evaluation, MSE, and gradients plus
 *  Hessian inverse update used in OBS.
 */


#include <vector>
#include <fcnn/level1.h>
#include <fcnn/level2.h>
#include <fcnn/level3.h>
#include <fcnn/report.h>
#include <fcnn/utils.h>
#include <fcnn/fcnncfg.h>
#if defined(HAVE_OPENMP)
#include <omp.h>
#endif /* defined(HAVE_OPENMP) */


using namespace fcnn::internal;



template <typename T>
void
fcnn::internal::eval(const int *lays, int no_lays, const int *n_pts,
                     const T *w_val, const int *af, const T *af_p,
                     int no_datarows, const T *in, T *out)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1];

#if defined(HAVE_OPENMP)
    std::vector<std::vector<T> > workv;
    T *work;
    int nth = 1, chsz = 1;
    #pragma omp parallel default(shared)
    {
    #pragma omp single
    {
        nth = omp_get_num_threads();
        if (nth > no_datarows) {
            nth = no_datarows;
            omp_set_num_threads(nth);
        } else {
            chsz = no_datarows / nth;
            if (no_datarows % nth) ++chsz;
        }
        for (int i = 0; i < nth; ++i) {
            workv.push_back(std::vector<T>(no_neurons));
        }
    }
#else /* defined(HAVE_OPENMP) */
    std::vector<T> workv(no_neurons);
    T *work = &workv[0];
#endif /* defined(HAVE_OPENMP) */
#if defined(HAVE_OPENMP)
    int i, ith;
    #pragma omp for schedule(static, chsz) private(i, ith, work)
    for (i = 0; i < no_datarows; ++i) {
#else /* defined(HAVE_OPENMP) */
    for (int i = 0; i < no_datarows; ++i) {
#endif /* defined(HAVE_OPENMP) */
#if defined(HAVE_OPENMP)
        ith = omp_get_thread_num();
        work = &workv[ith][0];
#endif /* defined(HAVE_OPENMP) */
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, af, af_p,
              work);
        // copy output
        copy(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows);
    }
#if defined(HAVE_OPENMP)
    } /* #pragma omp parallel */
#endif /* defined(HAVE_OPENMP) */
}



template <typename T>
T
fcnn::internal::mse(const int *lays, int no_lays, const int *n_pts,
                    const T *w_val, const int *af, const T *af_p,
                    int no_datarows, const T *in, const T *out)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1];

    T se = T();

#if defined(HAVE_OPENMP)
    std::vector<std::vector<T> > workv;
    T *work;
    int nth = 1, chsz = 1;
    #pragma omp parallel default(shared)
    {
    #pragma omp single
    {
        nth = omp_get_num_threads();
        if (nth > no_datarows) {
            nth = no_datarows;
            omp_set_num_threads(nth);
        } else {
            chsz = no_datarows / nth;
            if (no_datarows % nth) ++chsz;
        }
        for (int i = 0; i < nth; ++i) {
            workv.push_back(std::vector<T>(no_neurons));
        }
    }
#else /* defined(HAVE_OPENMP) */
    std::vector<T> workv(no_neurons);
    T *work = &workv[0];
#endif /* defined(HAVE_OPENMP) */
#if defined(HAVE_OPENMP)
    int i, ith;
    #pragma omp for schedule(static, chsz) \
        private(i, ith, work) \
        reduction(+:se)
    for (i = 0; i < no_datarows; ++i) {
#else /* defined(HAVE_OPENMP) */
    for (int i = 0; i < no_datarows; ++i) {
#endif /* defined(HAVE_OPENMP) */
#if defined(HAVE_OPENMP)
        ith = omp_get_thread_num();
        work = &workv[ith][0];
#endif /* defined(HAVE_OPENMP) */
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, af, af_p,
              work);
        // update se
        se += sumsqdiff(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows);
    }
#if defined(HAVE_OPENMP)
    } /* #pragma omp parallel */
#endif /* defined(HAVE_OPENMP) */

    return (T).5 * se / ((T)no_datarows * (T)no_outputs);
}



template <typename T>
T
fcnn::internal::grad(const int *lays, int no_lays, const int *n_pts,
                     const int *w_pts, const int *w_fl, const T *w_val,
                     const int *af, const T *af_p,
                     int no_datarows, const T *in, const T *out, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1],
        no_weights = w_pts[no_lays];
    T se = T();
#if defined(HAVE_OPENMP)
    std::vector<std::vector<T> > workv, deltav, gradv;
    T *work, *delta, *grad;
    int nth = 1, chsz = 1;
    #pragma omp parallel default(shared)
    {
    #pragma omp single
    {
        nth = omp_get_num_threads();
        if (nth > no_datarows) {
            nth = no_datarows;
            omp_set_num_threads(nth);
        } else {
            chsz = no_datarows / nth;
            if (no_datarows % nth) ++chsz;
        }
        for (int i = 0; i < nth; ++i) {
            workv.push_back(std::vector<T>(no_neurons));
            deltav.push_back(std::vector<T>(no_neurons));
            gradv.push_back(std::vector<T>(no_weights));
        }
    }
#else /* defined(HAVE_OPENMP) */
    std::vector<T> workv(no_neurons), deltav(no_neurons),
                   gradv(no_weights);
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
#endif /* defined(HAVE_OPENMP) */
    // loop over records
#if defined(HAVE_OPENMP)
    int i, ith;
    #pragma omp for schedule(static, chsz) \
        private(i, ith, work, delta, grad) \
        reduction(+:se)
    for (i = 0; i < no_datarows; ++i) {
#else /* defined(HAVE_OPENMP) */
    for (int i = 0; i < no_datarows; ++i) {
#endif /* defined(HAVE_OPENMP) */
#if defined(HAVE_OPENMP)
        ith = omp_get_thread_num();
        // set deltas to zero
        deltav[ith].assign(no_neurons, T());
        work = &workv[ith][0];
        delta = &deltav[ith][0];
        grad = &gradv[ith][0];
#else /* defined(HAVE_OPENMP) */
        // set deltas to zero
        deltav.assign(no_neurons, T());
#endif /* defined(HAVE_OPENMP) */
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, af, af_p,
              work);
        // init deltas
        diff(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows,
             delta + n_pts[no_lays - 1], 1);
        // update se
        se += sumsq(no_outputs, delta + n_pts[no_lays - 1], 1);
        // backpropagation
        backprop(lays, no_lays, n_pts,
                 no_weights, w_val, af, af_p,
                 work, delta, grad);
    }
#if defined(HAVE_OPENMP)
    } /* #pragma omp parallel */
    for (int th = 1; th < nth; ++th) {
        axpy(no_weights, 1., &gradv[th][0], 1, &gradv[0][0], 1);
    }
    grad = &gradv[0][0];
#endif /* defined(HAVE_OPENMP) */

    // get derivatives for active weights
    for (int i = 0, j = 0, n = no_weights; i < n; ++i)
        if (w_fl[i]) gr[j++] = grad[i] / ((T)no_datarows * (T)no_outputs);
    // scale mse and return
    return (T).5 * se / ((T)no_datarows * (T)no_outputs);
}




template <typename T>
void
fcnn::internal::gradi(const int *lays, int no_lays, const int *n_pts,
                      const int *w_pts, const int *w_fl, const T *w_val,
                      const int *af, const T *af_p,
                      int no_datarows, int i, const T *in, const T *out, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1],
        no_weights = w_pts[no_lays];

    std::vector<T> workv(no_neurons), deltav(no_neurons, T()),
                   gradv(no_weights, T());
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
    // copy input
    copy(no_inputs, in + i, no_datarows, work, 1);
    // feed forward
    feedf(lays, no_lays, n_pts,
          w_val, af, af_p,
          work);
    // init deltas
    diff(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows,
            delta + n_pts[no_lays - 1], 1);
    // backpropagation
    backprop(lays, no_lays, n_pts,
             no_weights, w_val, af, af_p,
             work, delta, grad);

    // get derivatives for active weights
    for (int ii = 0, j = 0, n = no_weights; ii < n; ++ii)
        if (w_fl[ii]) gr[j++] = grad[ii] / (T)no_outputs;
}




template <typename T>
void
fcnn::internal::gradij(const int *lays, int no_lays, const int *n_pts,
                       const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
                       const int *af, const T *af_p,
                       int no_datarows, int i, const T *in, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1],
        no_weights = w_pts[no_lays];

    std::vector<T> workv(no_neurons), deltav(no_neurons, T()),
                   gradv(no_weights, T());
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
    // loop over output neurons
    for (int j = 0; j < no_outputs; ++j) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, af, af_p,
              work);
        // init jth output neuron's delta
        delta[n_pts[no_lays - 1] + j] = 1;
        // backpropagation
        backpropj(lays, no_lays, n_pts, j,
                  w_pts, w_val, af, af_p,
                  work, delta, grad);
        // copy gradient
        for (int ii = 0, k = 0, n = no_weights; ii < n; ++ii)
            if (w_fl[ii]) gr[j * no_w_on + k++] = grad[ii];
        // set deltas and gradients to zero
        if (j < no_outputs - 1) {
            deltav.assign(no_neurons, T());
            gradv.assign(no_weights, T());
        }
    }
}




template <typename T>
void
fcnn::internal::jacob(const int *lays, int no_lays, const int *n_pts,
                      const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
                      const int *af, const T *af_p,
                      int no_datarows, int i, const T *in, T *jac)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1];

    std::vector<T> workv(no_neurons), deltav(no_neurons, T());
    T *work = &workv[0], *delta = &deltav[0];
    // loop over output neurons
    for (int j = 0; j < no_outputs; ++j) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, af, af_p,
              work);
        // init jth output neuron's delta
        delta[n_pts[no_lays - 1] + j] = 1;
        // backpropagation
        backpropjd(lays, no_lays, n_pts, j,
                   w_pts, w_val, af, af_p,
                   work, delta);
        // copy gradient
        for (int ii = 0; ii < no_inputs; ++ii)
             jac[j * no_inputs + ii] = delta[ii];
        // set deltas to zero
        if (j < no_outputs - 1) {
            deltav.assign(no_neurons, T());
        }
    }
}


#if defined(HAVE_BLAS)
extern "C" {
void
F77_FUNC(sgemv,SGEMV)(char *trans, int *M, int *N, float *alpha,
                      const float* A, int *lda,
                      const float* x, int *incx,
                      float *beta, float* y, int *incy);
void
F77_FUNC(dgemv,DGEMV)(char *trans, int *M, int *N, double *alpha,
                      const double* A, int *lda,
                      const double* x, int *incx,
                      double *beta, double* y, int *incy);
void
F77_FUNC(sger,SGER)(int *M, int *N, float *alpha,
                    const float *x, int *incx, const float *y, int *incy,
                    float *A, int *lda);
void
F77_FUNC(dger,DGER)(int *M, int *N, double *alpha,
                    const double *x, int *incx, const double *y, int *incy,
                    double *A, int *lda);
} /* extern "C" */


inline
void
gemv(char trans, int M, int N, float alpha,
     const float* A, int lda,
     const float* x, int incx,
     float beta, float* y, int incy)
{
    F77_FUNC(sgemv,SGEMV)(&trans, &M, &N,
                          &alpha, A, &lda, x, &incx,
                          &beta, y, &incy);
}


inline
void
gemv(char trans, int M, int N, double alpha,
     const double* A, int lda,
     const double* x, int incx,
     double beta, double* y, int incy)
{
    F77_FUNC(dgemv,DGEMV)(&trans, &M, &N,
                          &alpha, A, &lda, x, &incx,
                          &beta, y, &incy);
}


inline
void
ger(int M, int N, float alpha,
    const float *x, int incx, const float *y, int incy,
    float *A, int lda)
{
    F77_FUNC(sger,SGER)(&M, &N, &alpha,
                        x, &incx, y, &incy,
                        A, &lda);
}


inline
void
ger(int M, int N, double alpha,
    const double *x, int incx, const double *y, int incy,
    double *A, int lda)
{
    F77_FUNC(dger,DGER)(&M, &N, &alpha,
                        x, &incx, y, &incy,
                        A, &lda);
}



template <typename T>
void
fcnn::internal::ihessupdate(int nw, int no, T a, const T *g, T *H)
{
    std::vector<T> HXv(nw);
    T *HX = &HXv[0];
    const T *X = g;
    T alpha, one = 1., zero = 0.;
    for (int j = 0; j < no; ++j, X += nw) {
        gemv('N', nw, nw, one, H, nw, X, 1, zero, HX, 1);
        alpha = (T)-1. / (a + dot(nw, X, 1, HX, 1));
        ger(nw, nw, alpha, HX, 1, HX, 1, H, nw);
    }
}


#endif /* defined(HAVE_BLAS) */



// Explicit instantiations
#if !defined(FCNN_DOUBLE_ONLY)
template void fcnn::internal::eval(const int*, int, const int*,
                                   const float*, const int*, const float*,
                                   int, const float*, float*);
template float fcnn::internal::mse(const int*, int, const int*,
                                   const float*, const int*, const float*,
                                   int, const float*, const float*);
template float fcnn::internal::grad(const int*, int, const int*,
                                    const int*, const int*, const float*,
                                    const int*, const float*,
                                    int, const float*, const float*, float*);
template void fcnn::internal::gradi(const int*, int, const int*,
                                    const int*, const int*, const float*,
                                    const int*, const float*,
                                    int, int, const float*, const float*, float*);
template void fcnn::internal::gradij(const int*, int, const int*,
                                     const int*, const int*, const float*, int,
                                     const int*, const float*,
                                     int, int, const float*, float*);
template void fcnn::internal::jacob(const int*, int, const int*,
                                    const int*, const int*, const float*, int,
                                    const int*, const float*,
                                    int, int, const float*, float*);
#if defined(HAVE_BLAS)
template void fcnn::internal::ihessupdate(int, int, float, const float*, float*);
#endif /* defined(HAVE_BLAS) */
#endif /* !defined(FCNN_DOUBLE_ONLY) */
template void fcnn::internal::eval(const int*, int, const int*,
                                   const double*, const int*, const double*,
                                   int, const double*, double*);
template double fcnn::internal::mse(const int*, int, const int*,
                                    const double*, const int*, const double*,
                                    int, const double*, const double*);
template double fcnn::internal::grad(const int*, int, const int*,
                                     const int*, const int*, const double*,
                                     const int*, const double*,
                                     int, const double*, const double*, double*);
template void fcnn::internal::gradi(const int*, int, const int*,
                                    const int*, const int*, const double*,
                                    const int*, const double*,
                                    int, int, const double*, const double*, double*);
template void fcnn::internal::gradij(const int*, int, const int*,
                                     const int*, const int*, const double*, int,
                                     const int*, const double*,
                                     int, int, const double*, double*);
template void fcnn::internal::jacob(const int*, int, const int*,
                                    const int*, const int*, const double*, int,
                                    const int*, const double*,
                                    int, int, const double*, double*);
#if defined(HAVE_BLAS)
template void fcnn::internal::ihessupdate(int, int, double, const double*, double*);
#endif /* defined(HAVE_BLAS) */

