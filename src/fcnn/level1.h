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

/** \file level1.h
 *  \brief Level 1: BLAS1-like computational routines.
 */

#ifndef FCNN_LEVEL1_H

#define FCNN_LEVEL1_H


#include <fcnn/level1_impl.h>


namespace fcnn {
namespace internal {


/// Dot product
inline
float
dot_prod(int n, const float* x, const float* y)
{
    return DOT_PROD<float, 8>::eval(n, x, y);
}


/// Dot product
inline
double
dot_prod(int n, const double* x, const double* y)
{
    return DOT_PROD<double, 4>::eval(n, x, y);
}




/// Copy vector
inline
void
copy(int n, const float* x, int incx, float* y, int incy)
{
    COPY<float, 8>::copy(n, x, incx, y, incy);
}


/// Copy vector
inline
void
copy(int n, const double* x, int incx, double* y, int incy)
{
    COPY<double, 4>::copy(n, x, incx, y, incy);
}



/// BLAS1 axpy
inline
void
axpy(int n, const float &a, const float* x, int incx, float* y, int incy)
{
    AXPY<float, 8>::axpy(n, a, x, incx, y, incy);
}


/// BLAS1 axpy
inline
void
axpy(int n, const float &a, const double* x, int incx, double* y, int incy)
{
    AXPY<double, 4>::axpy(n, a, x, incx, y, incy);
}




/// Sum of squared differences
inline
float
sumsqerr(int n, const float* x, int incx, const float* y, int incy)
{
    return SUMSQERR<float, 8>::eval(n, x, incx, y, incy);
}


/// Sum of squared differences
inline
double
sumsqerr(int n, const double* x, int incx, const double* y, int incy)
{
    return SUMSQERR<double, 4>::eval(n, x, incx, y, incy);
}




} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL1_H */
