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

/** \file level2.h
 *  \brief Level 2 operations: single feed forward and backpropagation runs.
 */

#ifndef FCNN_LEVEL2_H

#define FCNN_LEVEL2_H


namespace fcnn {
namespace internal {


/// Feed forward - compute all neuron states based on states of neurons
/// in the input layers.
template <typename T>
void
feedf(const int *lays, int no_lays, const int *n_pts,
      const T *w_val, const int *af, const T *af_p,
      T *n_st);


/// Backpropagation - backpropagate errors in the output layer and determine
/// the MSE gradient (derivatives w.r.t weights).
template <typename T>
void
backprop(const int *lays, int no_lays, const int *n_pts,
         int no_weights, const T *w_val, const int *af, const T *af_p,
         const T *n_st, T *delta, T *grad);


/// Backpropagation - backpropagate error at the jth neuron the output layer
/// and determine the derivatives of jth output w.r.t weights (gradient).
template <typename T>
void
backpropj(const int *lays, int no_lays, const int *n_pts, int j,
          const int *w_pts, const T *w_val, const int *af, const T *af_p,
          const T *n_st, T *delta, T *grad);


/// Backpropagation - backpropagate error (delta) at the jth neuron the output layer
/// to input layers without computing derivatives w.r.t. weights.
template <typename T>
void
backpropjd(const int *lays, int no_lays, const int *n_pts, int j,
           const int *w_pts, const T *w_val, const int *af, const T *af_p,
           const T *n_st, T *delta);



} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL2_H */
