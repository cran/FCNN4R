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

/** \file struct.h
 *  \brief Working with multilayer perceptron network structure.
 */


#ifndef FCNN_STRUCT_H

#define FCNN_STRUCT_H


#include <vector>
#include <map>
#include <string>


namespace fcnn {
namespace internal {


/// Construct multilayer perceptron network given no. of neurons in layers.
template <typename T>
void mlp_construct(const std::vector<int> &layers,
                   std::vector<int> &n_p,
                   std::vector<int> &n_prev,
                   std::vector<int> &n_next,
                   std::vector<int> &w_p,
                   std::vector<T> &w_val,
                   std::vector<int> &w_fl,
                   int &w_on);


/// Construct multilayer perceptron network given no. of neurons in layers,
/// weights flags and active weights' values.
template <typename T>
void mlp_construct(const std::vector<int> &layers,
                   std::vector<int> &n_p,
                   std::vector<int> &n_prev,
                   std::vector<int> &n_next,
                   std::vector<int> &w_p,
                   const std::vector<T> &w_act_val,
                   std::vector<T> &w_val,
                   const std::vector<int> &w_fl,
                   int &w_on);


/// Reconstruct network by adding and/or reordeing input neurons. All new
/// connections are inactive. Requires the new total no. of input neurons
/// and a map of old inputs to new ones (1-based indexing).
template <typename T>
void mlp_expand_reorder_inputs(std::vector<int> &layers,
                               std::vector<int> &n_p,
                               std::vector<int> &n_prev,
                               std::vector<int> &n_next,
                               std::vector<int> &w_p,
                               std::vector<T> &w_val,
                               std::vector<int> &w_fl,
                               int newnoinp, const std::map<int, int> &m);


/// Reconstruct network by removing redundant neurons.
template <typename T>
int mlp_rm_neurons(std::vector<int> &layers,
                   std::vector<int> &n_p,
                   std::vector<int> &n_prev,
                   std::vector<int> &n_next,
                   std::vector<int> &w_p,
                   std::vector<T> &w_val,
                   std::vector<int> &w_fl,
                   int &w_on,
                   std::vector<int> & af,
                   std::vector<T> &af_p,
                   bool report);


/// Reconstruct network by removing redundant input neurons.
template <typename T>
void mlp_rm_input_neurons(std::vector<int> &layers,
                          std::vector<int> &n_p,
                          std::vector<int> &n_prev,
                          std::vector<int> &n_next,
                          std::vector<int> &w_p,
                          std::vector<T> &w_val,
                          std::vector<int> &w_fl,
                          bool report);


/// Merge two networks (they must have the same number of layers).
template <typename T>
void mlp_merge(const std::vector<int> &Alayers, const std::vector<int> &Aw_p,
               const std::vector<T> &Aw_val, const std::vector<int> &Aw_fl,
               const std::vector<int> &Blayers, const std::vector<int> &Bw_p,
               const std::vector<T> &Bw_val, const std::vector<int> &Bw_fl,
               bool same_inputs,
               std::vector<int> &layers, std::vector<int> &n_p,
               std::vector<int> &n_prev, std::vector<int> &n_next,
               std::vector<int> &w_p, std::vector<T> &w_val,
               std::vector<int> &w_fl, int &w_on);


/// Connect one network outputs to another network inputs (the numbers of output
/// and input neurons must agree).
template <typename T>
void mlp_stack(const std::vector<int> &Alayers, const std::vector<int> &Aw_p,
               const std::vector<T> &Aw_val, const std::vector<int> &Aw_fl,
               const std::vector<int> &Blayers, const std::vector<int> &Bw_p,
               const std::vector<T> &Bw_val, const std::vector<int> &Bw_fl,
               std::vector<int> &layers, std::vector<int> &n_p,
               std::vector<int> &n_prev, std::vector<int> &n_next,
               std::vector<int> &w_p, std::vector<T> &w_val,
               std::vector<int> &w_fl, int &w_on);




/// Save network in a text file.
template <typename T>
bool mlp_save_txt(const std::string &fname,
                  const std::string &netname,
                  const std::vector<int> &layers,
                  const std::vector<T> &w_val,
                  const std::vector<int> &w_fl,
                  const std::vector<int> &af,
                  const std::vector<T> &af_p);


/// Load network in a text file.
template <typename T>
bool mlp_load_txt(const std::string &fname,
                  std::string &netname,
                  std::vector<int> &layers,
                  std::vector<int> &n_p,
                  std::vector<int> &n_prev,
                  std::vector<int> &n_next,
                  std::vector<int> &w_p,
                  std::vector<T> &w_val,
                  std::vector<int> &w_fl,
                  int &w_on,
                  std::vector<int> &af,
                  std::vector<T> &af_p);


/// Get absolute neuron index given layer and neuron index within this layer.
inline
int
mlp_get_n_idx(const int *n_p, int l, int n)
{
    return n_p[l - 1] + n - 1;
}


/// Get absolute weight index given layer, neuron index within this layer,
/// and index of neuron in the previous layer.
inline
int
mlp_get_w_idx(const int *layers, const int *w_p,
              int l, int n, int npl)
{
    return w_p[l - 1] + (n - 1) * (layers[l - 2] + 1) + npl;
}

/// Get layer, neuron index within this layer, and index of neuron
/// in the previous layer given absolute weight index.
void mlp_get_lnn_idx(const int *layers, const int *w_p,
                     int i, int &l, int &n, int &npl);


/// Get absolute weight index given index within active ones.
int mlp_get_abs_w_idx(const int *w_fl, int i);


/// Set weight (in)active and update network structure data.
template <typename T>
void mlp_set_active(const int *layers, const int *n_p, int *n_prev, int *n_next,
                    const int *w_p, T *w_val, int *w_fl, int *w_on,
                    int l, int n, int npl, bool on);

/// Set weight (in)active and update network structure data.
template <typename T>
void mlp_set_active(const int *layers, const int *n_p, int *n_prev, int *n_next,
                    const int *w_p, T *w_val, int *w_fl, int *w_on,
                    int i, bool on);



} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_STRUCT_H */
