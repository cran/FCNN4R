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

/** \file level3.h
 *  \brief Level 3 operations: evaluation, MSE, and gradients plus
 *  Hessian inverse update used in OBS.
 */

#ifndef FCNN_LEVEL3_H

#define FCNN_LEVEL3_H


namespace fcnn {
namespace internal {


/// Evaluate network output given input
template <typename T>
void
eval(const int *lays, int no_lays, const int *n_pts,
     const T *w_val, const int *af, const T *af_p,
     int no_datarows, const T *in, T *out);


/// Determine network's MSE given input and expected output.
template <typename T>
T
mse(const int *lays, int no_lays, const int *n_pts,
    const T *w_val, const int *af, const T *af_p,
    int no_datarows, const T *in, const T *out);


/// Compute gradient of MSE (derivatives w.r.t. active weights)
/// given input and expected output.
template <typename T>
T
grad(const int *lays, int no_lays, const int *n_pts,
     const int *w_pts, const int *w_fl, const T *w_val,
     const int *af, const T *af_p,
     int no_datarows, const T *in, const T *out, T *gr);

/// Compute gradient of MSE (derivatives w.r.t. active weights)
/// given input and expected output using ith row of data only. This is
/// normalised by the number of outputs only.
template <typename T>
void
gradi(const int *lays, int no_lays, const int *n_pts,
      const int *w_pts, const int *w_fl, const T *w_val,
      const int *af, const T *af_p,
      int no_datarows, int i, const T *in, const T *out, T *gr);

/// Compute gradients of networks outputs, i.e the derivatives of outputs
/// w.r.t. active weights, at given data row.
template <typename T>
void
gradij(const int *lays, int no_lays, const int *n_pts,
       const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
       const int *af, const T *af_p,
       int no_datarows, int i, const T *in, T *gr);

/// Compute the Jacobian of network transformation, i.e the derivatives
/// of outputs w.r.t. network inputs, at given data row.
template <typename T>
void
jacob(const int *lays, int no_lays, const int *n_pts,
      const int *w_pts, const int *w_fl, const T *w_val, int no_w_on,
      const int *af, const T *af_p,
      int no_datarows, int i, const T *in, T *jac);

/// Update Hessian inverse approximation given result from gradij.
/// Implemented only if BLAS library is present.
template <typename T>
void
ihessupdate(int nw, int no, T a, const T *g, T *Hinv);



} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL3_H */
