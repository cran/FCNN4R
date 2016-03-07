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

/** \file activation.h
 *  \brief Activation functions.
 */

#ifndef FCNN_ACTIVATION_H

#define FCNN_ACTIVATION_H


#include <cmath>
#include <string>
#include <fcnn/error.h>



namespace fcnn {


/// Multilayer perceptron activation functions enum.
enum mlp_activation_f {

    threshold = 1, ///< Threshold.
    sym_threshold, ///< Symmetric threshold.
    linear, ///< Linear \f$ f(x) = s x \f$.
    sigmoid, ///< Sigmoid \f$ f(x) = (1+\exp(-2 s x))^{-1} \f$.
    sym_sigmoid, ///< Symmetric sigmoid (tanh) \f$ f(x) = 2(1+\exp(-2 s x))^{-1} - 1 \f$.
    sigmoid_approx, ///< Sigmoid piecewise approximation.
    sym_sigmoid_approx ///< Symmetric sigmoid piecewise approximation.

}; /* enum mpl_activation_f */


namespace internal {



/// Check validity of activation function index.
inline bool
mlp_act_f_valid(int af)
{
    switch (af) {
        case threshold:
        case sym_threshold:
        case linear:
        case sigmoid:
        case sym_sigmoid:
        case sigmoid_approx:
        case sym_sigmoid_approx:
            return true;
        default:
            return false;
    }
}


/// Get default activation function parameter.
template <typename T>
inline T
mlp_act_f_pdefault(int af)
{
    switch (af) {
        case threshold:
        case sym_threshold:
        case linear:
            return (T)1;
        case sigmoid:
        case sym_sigmoid:
        case sigmoid_approx:
        case sym_sigmoid_approx:
            return (T).5;
        default:
            throw exception("invalid activation function id");
    }
}


/// Return string describing activation function.
inline std::string
mlp_act_f_str(int af)
{
    switch (af) {
        case threshold:
            return "threshold: f(x) = 1 : x >= 0, f(x) = 1 otherwise";
        case sym_threshold:
            return "symmetric threshold: f(x) = 1 : x >= 0, f(x) = -1 otherwise";
        case linear:
            return "linear: f(x) = s * x";
        case sigmoid:
            return "sigmoid: f(x) = (1 + exp(-2 * s * x)) ^ -1";
        case sym_sigmoid:
            return "symmetric sigmoid (tanh): f(x) = 2 * (1 + exp(-2 * s * x))^-1 - 1";
        case sigmoid_approx:
            return "sigmoid approx.: f(x) ~ (1 + exp(-2 * s * x))^-1";
        case sym_sigmoid_approx:
            return "sym. sigmoid approx.: f(x) ~ 2*(1 + exp(-2 * s * x))^-1 - 1";
        default:
            throw exception("invalid activation function id");
    }
    return "";
}



/// Hyperbolic tangent approximation.
template <typename T>
inline T
tanh_app(const T &x)
{
    if (x < (T) -2.8) return (T)2. / ((T)1. + std::exp((T)-2. * x)) - (T)1.;
    else if (x > (T) 2.8) return (T) (T)2. / ((T)1. + std::exp((T)-2. * x)) - (T)1.;
    else if (x < (T) -2.0) return (T) 0.03575493 * x - (T) 0.8925177;
    else if (x < (T) -1.2) return (T) 0.16296622 * x - (T) 0.6380951;
    else if (x < (T) -0.7) return (T) 0.45857366 * x - (T) 0.2833662;
    else if (x < (T) 0.7) return (T) 0.86338254 * x;
    else if (x < (T) 1.2) return (T) 0.45857366 * x + (T) 0.2833662;
    else if (x < (T) 2.0) return (T) 0.16296622 * x + (T) 0.6380951;
    return (T) 0.03575493 * x + (T) 0.8925177;
}



/// Evaluate activation function given slope parameter and argument value.
template <typename T>
inline T
mlp_act_f(int af, const T &s, const T &x)
{
    switch (af) {
        case threshold:
            return (x < (T) 0.) ? (T) 0. : (T) 1.;
        case sym_threshold:
            return (x < (T) 0.) ? (T) -1. : (T) 1.;
        case linear:
            return s * x;
        case sigmoid:
            return (T) 1. / ((T) 1. + std::exp((T) -2. * s * x));
        case sym_sigmoid:
            return (T) 2. / ((T) 1. + std::exp((T) -2. * s * x)) - (T) 1.;
        case sigmoid_approx:
            return (T) 0.5 + (T) 0.5 * tanh_app(s * x);
        case sym_sigmoid_approx:
            return tanh_app(s * x);
        default:
            throw exception("invalid activation function id");
    }
    return 0;
}


/// Evaluate the derivative of activation function given slope parameter
/// and function value.
template <typename T>
inline T
mlp_act_f_der(int af, const T &s, const T &y)
{
    switch (af) {
        case linear:
            return s;
        case sigmoid_approx:
        case sigmoid:
            return (T) 2. * s * y * ((T) 1. - y);
        case sym_sigmoid_approx:
        case sym_sigmoid:
            return s * ((T) 1. - y * y);
        case threshold:
        case sym_threshold:
            throw exception("trying to differentiate step function");
        default:
            throw exception("invalid activation function id");
    }
    return 0;
}



} /* namespace internal */
} /* namespace fcnn */



#endif /* FCNN_ACTIVATION_H */
