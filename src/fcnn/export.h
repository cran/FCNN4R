/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2015-2016
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

/** \file export.h
 *  \brief Exporting networks.
 */


#ifndef FCNN_EXPORT_H

#define FCNN_EXPORT_H


#include <vector>
#include <string>


namespace fcnn {
namespace internal {


/// Export trained network to a C function with optional input and output
/// affine transformations of the form \f$Ax+b\f$ (input)
/// and \f$Cx+d\f$ (output). When with_wp is set to true the backpropagation code
/// (for online learning) is also exported. In order to export the backpropagation
/// code, when output transformation \f$Cx+d\f$ is provided, one has to provide
/// the inverse transformation given by \f$Ex+f\f$.
template <typename T>
bool mlp_export_C(const std::string &fname,
                  const std::string &netname,
                  const std::vector<int> &layers,
                  const std::vector<int> &n_p,
                  const std::vector<T> &w_val,
                  const std::vector<int> &w_fl,
                  int w_on,
                  const std::vector<int> &af,
                  const std::vector<T> &af_p,
                  bool with_bp,
                  const T *A, const T *b,
                  const T *C, const T *d,
                  const T *E, const T *f);



} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_EXPORT_H */
