/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2012-2015
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
 *  \brief Level 3 operations: evaluation, MSE, and gradients.
 */


#include <vector>
#include <fcnn/level1.h>
#include <fcnn/level2.h>
#include <fcnn/level3.h>


using namespace fcnn::internal;



template <typename T>
void
fcnn::internal::eval(const int *lays, int no_lays, const int *n_pts,
                     const T *w_val, int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                     int no_datarows, const T *in, T *out)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1];

    std::vector<T> workv(no_neurons);
    T *work = &workv[0];

    for (int i = 0; i < no_datarows; ++i) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, hl_af, hl_af_p, ol_af, ol_af_p,
              work);
        // copy output
        copy(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows);
    }
}



template <typename T>
T
fcnn::internal::mse(const int *lays, int no_lays, const int *n_pts,
                    const T *w_val, int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                    int no_datarows, const T *in, const T *out)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1];

    T se = T();

    std::vector<T> workv(no_neurons);
    T *work = &workv[0];

    for (int i = 0; i < no_datarows; ++i) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, hl_af, hl_af_p, ol_af, ol_af_p,
              work);
        // update se
        se += sumsqerr(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows);
    }

    return (T).5 * se / ((T)no_datarows * (T)no_outputs);
}




template <typename T>
T
fcnn::internal::grad(const int *lays, int no_lays, const int *n_pts,
                     const int *w_pts, const int *w_fl, const T *w_val,
                     int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                     int no_datarows, const T *in, const T *out, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1],
        no_weights = w_pts[no_lays];

    T se = T();
    std::vector<T> workv(no_neurons),
                   deltav(no_weights, T()), gradv(no_weights, T());
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
    // loop over records
    for (int i = 0; i < no_datarows; ++i) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, hl_af, hl_af_p, ol_af, ol_af_p,
              work);
        // update se
        se += sumsqerr(no_outputs, work + n_pts[no_lays - 1], 1, out + i, no_datarows);
        // init deltas
        copy(no_outputs, work + n_pts[no_lays - 1], 1, delta + n_pts[no_lays - 1], 1);
        axpy(no_outputs, -1., out + i, no_datarows, delta + n_pts[no_lays - 1], 1);
        // backpropagation
        backprop(lays, no_lays, n_pts,
                 no_weights, w_val, hl_af, hl_af_p, ol_af, ol_af_p,
                 work, delta, grad);
        // set deltas to zero
        if (i != (no_datarows - 1)) deltav.assign(no_weights, T());
    }

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
                      int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                      int no_datarows, int i, const T *in, const T *out, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_outputs = lays[no_lays - 1],
        no_weights = w_pts[no_lays];

    std::vector<T> workv(no_neurons),
                   deltav(no_weights, T()), gradv(no_weights, T());
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
    // copy input
    copy(no_inputs, in + i, no_datarows, work, 1);
    // feed forward
    feedf(lays, no_lays, n_pts,
          w_val, hl_af, hl_af_p, ol_af, ol_af_p,
          work);
    // init deltas
    copy(no_outputs, work + n_pts[no_lays - 1], 1, delta + n_pts[no_lays - 1], 1);
    axpy(no_outputs, -1., out + i, no_datarows, delta + n_pts[no_lays - 1], 1);
    // backpropagation
    backprop(lays, no_lays, n_pts,
             no_weights, w_val, hl_af, hl_af_p, ol_af, ol_af_p,
             work, delta, grad);

    // get derivatives for active weights
    for (int ii = 0, j = 0, n = no_weights; ii < n; ++ii)
        if (w_fl[ii]) gr[j++] = grad[ii] / (T)no_outputs;
}




template <typename T>
void
fcnn::internal::gradij(const int *lays, int no_lays, const int *n_pts,
                       const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
                       int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                       int no_datarows, int i, const T *in, T *gr)
{
    int no_neurons = n_pts[no_lays],
        no_inputs = lays[0],
        no_weights = w_pts[no_lays];

    std::vector<T> workv(no_neurons),
                   deltav(no_weights, T()), gradv(no_weights, T());
    T *work = &workv[0], *delta = &deltav[0], *grad = &gradv[0];
    // loop over output neurons
    for (int j = 0; j < lays[no_lays - 1]; ++j) {
        // copy input
        copy(no_inputs, in + i, no_datarows, work, 1);
        // feed forward
        feedf(lays, no_lays, n_pts,
              w_val, hl_af, hl_af_p, ol_af, ol_af_p,
              work);
        // init jth output neuron's delta
        delta[n_pts[no_lays - 1] + j] = 1;
        // backpropagation
        backpropj(lays, no_lays, n_pts, j,
                  w_pts, w_val, hl_af, hl_af_p, ol_af, ol_af_p,
                  work, delta, grad);
        // copy gradient
        for (int ii = 0, k = 0, n = no_weights; ii < n; ++ii)
            if (w_fl[ii]) gr[j * no_w_on + k++] = grad[ii];
        // set deltas and gradients to zero
        if (j < lays[no_lays - 1] - 1) {
            deltav.assign(no_weights, T());
            gradv.assign(no_weights, T());
        }
    }
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::eval(const int*, int, const int*,
                                   const float*, int, float, int, float,
                                   int, const float*, float*);
template float fcnn::internal::mse(const int*, int, const int*,
                                   const float*, int, float, int, float,
                                   int, const float*, const float*);
template float fcnn::internal::grad(const int*, int, const int*,
                                    const int*, const int*, const float*,
                                    int, float, int, float,
                                    int, const float*, const float*, float*);
template void fcnn::internal::gradi(const int*, int, const int*,
                                    const int*, const int*, const float*,
                                    int, float, int, float,
                                    int, int, const float*, const float*, float*);
template void fcnn::internal::gradij(const int*, int, const int*,
                                     const int*, const int*, const float*, int,
                                     int, float, int, float,
                                     int, int, const float*, float*);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::eval(const int*, int, const int*,
                                   const double*, int, double, int, double,
                                   int, const double*, double*);
template double fcnn::internal::mse(const int*, int, const int*,
                                    const double*, int, double, int, double,
                                    int, const double*, const double*);
template double fcnn::internal::grad(const int*, int, const int*,
                                     const int*, const int*, const double*,
                                     int, double, int, double,
                                     int, const double*, const double*, double*);
template void fcnn::internal::gradi(const int*, int, const int*,
                                    const int*, const int*, const double*,
                                    int, double, int, double,
                                    int, int, const double*, const double*, double*);
template void fcnn::internal::gradij(const int*, int, const int*,
                                     const int*, const int*, const double*, int,
                                     int, double, int, double,
                                     int, int, const double*, double*);

