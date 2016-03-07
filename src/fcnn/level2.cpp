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

/** \file level2.cpp
 *  \brief Level 2 operations: single feed forward and backpropagation runs.
 */


#include <fcnn/activation.h>
#include <fcnn/level1.h>
#include <fcnn/level2.h>


using namespace fcnn::internal;


template <typename T>
void
fcnn::internal::feedf(const int *lays, int no_lays, const int *n_pts,
                      const T *w_val, const int *af, const T *af_p,
                      T *n_st)
{
    int wi = 0, ni = n_pts[1], l = 1, L = no_lays - 1;
    int actf; T actfp;
    // hidden layers
    for (; l < L; ++l) {
        int npl = lays[l - 1], nn = n_pts[l + 1];
        T *nplptr = &n_st[n_pts[l - 1]];
        actf = af[l]; actfp = af_p[l];
        for (; ni < nn; ++ni, wi += npl) {
            // bias
            T d = w_val[wi++];
            // dot product
            d += dot(npl, nplptr, 1, w_val + wi, 1);
            // activation
            n_st[ni] = mlp_act_f(actf, actfp, d);
        }
    }

    // output layer
    int npl = lays[l - 1], nn = n_pts[l + 1];
    T *nplptr = &n_st[n_pts[l - 1]];
    actf = af[l]; actfp = af_p[l];
    for (; ni < nn; ++ni, wi += npl) {
        // bias
        T d = w_val[wi++];
        // dot product
        d += dot(npl, nplptr, 1, w_val + wi, 1);
        // activation
        n_st[ni] = mlp_act_f(actf, actfp, d);
    }
}



template <typename T>
void
fcnn::internal::backprop(const int *lays, int no_lays, const int *n_pts,
                         int no_weights, const T *w_val,
                         const int *af, const T *af_p,
                         const T *n_st, T *delta, T *grad)
{
    // initialisation
    int l = no_lays - 1, ni = n_pts[no_lays] - 1, wi = no_weights, nlpl;
    register T d;
    int actf; T actfp;
    // output and hidden layers except for the 1st
    for (; l > 1; --l) {
        nlpl = lays[l - 1];
        actf = af[l]; actfp = af_p[l];
        for (int nl = lays[l]; nl; --nl, --ni) {
            d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
            wi -= nlpl;
            axpy(nlpl, d, w_val + wi, 1, delta + n_pts[l - 1], 1);
            axpy(nlpl, d, n_st + n_pts[l - 1], 1, grad + wi, 1);
            grad[--wi] += d;
        }
    }
    // first hidden layer
    nlpl = lays[0];
    actf = af[l]; actfp = af_p[l];
    for (int nl = lays[l]; nl; --nl, --ni) {
        d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
        wi -= nlpl;
        axpy(nlpl, d, n_st + n_pts[l - 1], 1, grad + wi, 1);
        grad[--wi] += d;
    }
}



template <typename T>
void
fcnn::internal::backpropj(const int *lays, int no_lays, const int *n_pts, int j,
                          const int *w_pts, const T *w_val,
                          const int *af, const T *af_p,
                          const T *n_st, T *delta, T *grad)
{
    // initialisation
    int l = no_lays - 1, ni = n_pts[l] + j,
        wi = w_pts[l] + j * (1 + lays[l - 1]) + 1, nlpl;
    register T d;
    int actf; T actfp;
    // jth output neuron delta
    actf = af[l]; actfp = af_p[l];
    d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
    nlpl = lays[l - 1];
    axpy(nlpl, d, w_val + wi, 1, delta + n_pts[l - 1], 1);
    axpy(nlpl, d, n_st + n_pts[l - 1], 1, grad + wi, 1);
    grad[--wi] += d;
    // hidden layers except for the 1st
    wi = w_pts[l];
    ni = n_pts[l] - 1;
    for (--l; l > 1; --l) {
        nlpl = lays[l - 1];
        actf = af[l]; actfp = af_p[l];
        for (int nl = lays[l]; nl; --nl, --ni) {
            d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
            wi -= nlpl;
            axpy(nlpl, d, w_val + wi, 1, delta + n_pts[l - 1], 1);
            axpy(nlpl, d, n_st + n_pts[l - 1], 1, grad + wi, 1);
            grad[--wi] += d;
        }
    }
    // first hidden layer
    nlpl = lays[0];
    actf = af[l]; actfp = af_p[l];
    for (int nl = lays[l]; nl; --nl, --ni) {
        d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
        wi -= nlpl;
        axpy(nlpl, d, n_st + n_pts[l - 1], 1, grad + wi, 1);
        grad[--wi] += d;
    }
}



template <typename T>
void
fcnn::internal::backpropjd(const int *lays, int no_lays, const int *n_pts, int j,
                           const int *w_pts, const T *w_val,
                           const int *af, const T *af_p,
                           const T *n_st, T *delta)
{
    // initialisation
    int l = no_lays - 1, ni = n_pts[l] + j,
        wi = w_pts[l] + j * (1 + lays[l - 1]) + 1, nlpl;
    register T d;
    int actf; T actfp;
    // jth output neuron delta
    actf = af[l]; actfp = af_p[l];
    d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
    nlpl = lays[l - 1];
    axpy(nlpl, d, w_val + wi, 1, delta + n_pts[l - 1], 1);
    // hidden layers
    wi = w_pts[l];
    ni = n_pts[l] - 1;
    for (--l; l; --l) {
        nlpl = lays[l - 1];
        actf = af[l]; actfp = af_p[l];
        for (int nl = lays[l]; nl; --nl, --ni) {
            d = delta[ni] * mlp_act_f_der(actf, actfp, n_st[ni]);
            wi -= nlpl;
            axpy(nlpl, d, w_val + wi, 1, delta + n_pts[l - 1], 1);
            --wi;
        }
    }
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::feedf(const int*, int, const int*,
                                    const float*, const int*, const float*,
                                    float*);
template void fcnn::internal::backprop(const int*, int, const int*,
                                       int, const float*, const int*, const float*,
                                       const float*, float*, float*);
template void fcnn::internal::backpropj(const int*, int, const int*, int,
                                        const int*, const float*, const int*, const float*,
                                        const float*, float*, float*);
template void fcnn::internal::backpropjd(const int*, int, const int*, int,
                                         const int*, const float*, const int*, const float*,
                                         const float*, float*);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::feedf(const int*, int, const int*,
                                    const double*, const int*, const double*,
                                    double*);
template void fcnn::internal::backprop(const int*, int, const int*,
                                       int, const double*, const int*, const double*,
                                       const double*, double*, double*);
template void fcnn::internal::backpropj(const int*, int, const int*, int,
                                        const int*, const double*, const int*, const double*,
                                        const double*, double*, double*);
template void fcnn::internal::backpropjd(const int*, int, const int*, int,
                                         const int*, const double*, const int*, const double*,
                                         const double*, double*);

