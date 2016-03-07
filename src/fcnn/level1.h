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

/** \file level1.h
 *  \brief Level 1: BLAS1-like computational routines.
 */

#ifndef FCNN_LEVEL1_H

#define FCNN_LEVEL1_H


#include <fcnn/level1_impl.h>


namespace fcnn {
namespace internal {


// NOTE: We are not using BLAS1 here - usually BLAS implementations focus
// on the speed of level 2 and level 3 routines, benchmarks show that our
// implementation is faster than those found in most of BLAS libs avaialable.
#if 0 // defined(HAVE_BLAS)
extern "C" {
float
F77_FUNC(sdot,SDOT)(const int *n,
                    const float *dx, const int *incx,
                    const float *dy, const int *incy);
double
F77_FUNC(ddot,DDOT)(const int *n,
                    const double *dx, const int *incx,
                    const double *dy, const int *incy);
void
F77_FUNC(scopy,SCOPY)(const int *n,
                      const float *dx, const int *incx,
                      float *dy, const int *incy);
void
F77_FUNC(dcopy,DCOPY)(const int *n,
                      const double *dx, const int *incx,
                      double *dy, const int *incy);
void
F77_FUNC(saxpy,SAXPY)(const int *n, const float *da,
                      const float *dx, const int *incx,
                      float *dy, const int *incy);
void
F77_FUNC(daxpy,DAXPY)(const int *n, const double *da,
                      const double *dx, const int *incx,
                      double *dy, const int *incy);
}
#endif


/// BLAS1 dot product
inline
float
dot(int n, const float* x, int incx, const float* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    return F77_FUNC(sdot,SDOT)(&n, x, &incx, y, &incy);
#else
    return DOT<float, 4>::dot(n, x, incx, y, incy);
#endif
}


/// BLAS1 dot product
inline
double
dot(int n, const double* x, int incx, const double* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    return F77_FUNC(ddot,DDOT)(&n, x, &incx, y, &incy);
#else
    return DOT<double, 4>::dot(n, x, incx, y, incy);
#endif
}




/// BLAS1 copy
inline
void
copy(int n, const float* x, int incx, float* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    F77_FUNC(scopy,SCOPY)(&n, x, &incx, y, &incy);
#else
    COPY<float, 4>::copy(n, x, incx, y, incy);
#endif
}


/// BLAS1 copy
inline
void
copy(int n, const double* x, int incx, double* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    F77_FUNC(dcopy,DCOPY)(&n, x, &incx, y, &incy);
#else
    COPY<double, 4>::copy(n, x, incx, y, incy);
#endif
}





/// BLAS1 axpy
inline
void
axpy(int n, const float &a, const float* x, int incx, float* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    F77_FUNC(saxpy,SAXPY)(&n, &a, x, &incx, y, &incy);
#else
    AXPY<float, 4>::axpy(n, a, x, incx, y, incy);
#endif
}


/// BLAS1 axpy
inline
void
axpy(int n, const double &a, const double* x, int incx, double* y, int incy)
{
#if 0 // defined(HAVE_BLAS)
    F77_FUNC(daxpy,DAXPY)(&n, &a, x, &incx, y, &incy);
#else
    AXPY<double, 4>::axpy(n, a, x, incx, y, incy);
#endif
}




/// Vector difference, x - y -> z
inline
void
diff(int n, const float* x, int incx, const float* y, int incy, float* z, int incz)
{
    return DIFF<float, 4>::diff(n, x, incx, y, incy, z, incz);
}


/// Vector difference, x - y -> z
inline
void
diff(int n, const double* x, int incx, const double* y, int incy, double* z, int incz)
{
    return DIFF<double, 4>::diff(n, x, incx, y, incy, z, incz);
}




/// Sum of squared differences
inline
float
sumsqdiff(int n, const float* x, int incx, const float* y, int incy)
{
    return SUMSQDIFF<float, 4>::sumsqdiff(n, x, incx, y, incy);
}


/// Sum of squared differences
inline
double
sumsqdiff(int n, const double* x, int incx, const double* y, int incy)
{
    return SUMSQDIFF<double, 4>::sumsqdiff(n, x, incx, y, incy);
}




/// Sum of squares
inline
float
sumsq(int n, const float* x, int incx)
{
    return SUMSQ<float, 4>::sumsq(n, x, incx);
}


/// Sum of squares
inline
double
sumsq(int n, const double* x, int incx)
{
    return SUMSQ<double, 4>::sumsq(n, x, incx);
}




} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL1_H */
