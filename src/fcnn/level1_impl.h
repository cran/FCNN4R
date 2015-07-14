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

/** \file level1_impl.h
 *  \brief BLAS1-like routines implementation.
 */

#ifndef FCNN_LEVEL1_IMPL_H

#define FCNN_LEVEL1_IMPL_H


namespace fcnn {
namespace internal {


template <typename T, int N>
struct DOT {
    static inline T eval(const T* x, const T* y) {
        return *x * *y + DOT<T, N - 1>::eval(x + 1, y + 1);
    }
};


template <typename T>
struct DOT<T, 1> {
    static inline T eval(const T* x, const T* y) {
        return *x * *y;
    }
};

template <typename T, int N>
struct DOT_SWITCH {
    static inline T eval(int n, const T* x, const T* y) {
        if (n == N) return DOT<T, N>::eval(x, y);
        else return DOT_SWITCH<T, N - 1>::eval(n, x, y);
    }
};

template <typename T>
struct DOT_SWITCH<T, 0> {
    static inline T eval(int n, const T* x, const T* y) {
        return T();
    }
};

template <typename T, int N>
struct DOT_UNROLL {
    static inline T eval(int k, const T* x, const T* y) {
        T s = T();
        for (int i = 0; i < k; ++i, x += N, y += N)
            s += DOT<T, N>::eval(x, y);
        return s;
    }
};

template <typename T, int N>
struct DOT_PROD {
    static inline T eval(int n, const T* x, const T* y) {
        if (n > N)  {
            int k = n / N, kr = n % N;
            return DOT_UNROLL<T, N>::eval(k, x, y)
                   + DOT_SWITCH<T, N>::eval(kr, x + n - kr, y + n - kr);
        } else return DOT_SWITCH<T, N>::eval(n, x, y);
    }
};






template <typename T, int N>
struct COPY_BLOCK {
    static inline void copy(const T* x, int incx, T* y, int incy) {
        *y = *x;
        COPY_BLOCK<T, N - 1>::copy(x + incx, incx, y + incy, incy);
    }
};

template <typename T>
struct COPY_BLOCK<T, 1> {
    static inline void copy(const T* x, int incx, T* y, int incy) {
        *y = *x;
    }
};

template <typename T, int N>
struct COPY_SWITCH {
    static inline void copy(int n, const T* x, int incx, T* y, int incy) {
        if (n == N) COPY_BLOCK<T, N>::copy(x, incx, y, incy); else
        COPY_SWITCH<T, N - 1>::copy(n, x, incx, y, incy);
    }
};

template <typename T>
struct COPY_SWITCH<T, 0> {
    static inline void copy(int n, const T* x, int incx, T* y, int incy) {
        ;
    }
};

template <typename T, int N>
struct COPY_UNROLL {
    static inline void copy(int k, const T* x, int incx, T* y, int incy) {
        int incaN = incx * N, incbN = incy * N;
        for (int i = 0; i < k; ++i, x += incaN, y += incbN)
            COPY_BLOCK<T, N>::copy(x, incx, y, incy);
    }
};

template <typename T, int N>
struct COPY {
    static inline void copy(int n, const T* x, int incx, T* y, int incy) {
        if (n > N) {
            int k = n / N, kr = n % N;
            COPY_UNROLL<T, N>::copy(k, x, incx, y, incy);
            COPY_SWITCH<T, N>::copy(kr, x + (n - kr) * incx, incx, y + (n - kr) * incy, incy);
        } else COPY_SWITCH<T, N>::copy(n, x, incx, y, incy);
    }
};






template <typename T, int N>
struct AXPY_BLOCK {
    static inline void axpy(const T &a, const T* x, int incx, T* y, int incy) {
        *y += a * *x;
        AXPY_BLOCK<T, N - 1>::axpy(a, x + incx, incx, y + incy, incy);
    }
};

template <typename T>
struct AXPY_BLOCK<T, 1> {
    static inline void axpy(const T &a, const T* x, int incx, T* y, int incy) {
        *y += a * *x;
    }
};

template <typename T, int N>
struct AXPY_SWITCH {
    static inline void axpy(int n, const T &a, const T* x, int incx, T* y, int incy) {
        if (n == N) AXPY_BLOCK<T, N>::axpy(a, x, incx, y, incy); else
        AXPY_SWITCH<T, N - 1>::axpy(n, a, x, incx, y, incy);
    }
};

template <typename T>
struct AXPY_SWITCH<T, 0> {
    static inline void axpy(int n, const T &a, const T* x, int incx, T* y, int incy) {
        ;
    }
};

template <typename T, int N>
struct AXPY_UNROLL {
    static inline void axpy(int k, const T &a, const T* x, int incx, T* y, int incy) {
        int incxN = incx * N, incyN = incy * N;
        for (int i = 0; i < k; ++i, x += incxN, y += incyN)
            AXPY_BLOCK<T, N>::axpy(a, x, incx, y, incy);
    }
};

template <typename T, int N>
struct AXPY {
    static inline void axpy(int n, const T &a, const T* x, int incx, T* y, int incy) {
        if (n > N) {
            int k = n / N, kr = n % N;
            AXPY_UNROLL<T, N>::axpy(k, a, x, incx, y, incy);
            AXPY_SWITCH<T, N>::axpy(kr, a, x + (n - kr) * incx, incx, y + (n - kr) * incy, incy);
        } else AXPY_SWITCH<T, N>::axpy(n, a, x, incx, y, incy);
    }
};







template <typename T, int N>
struct SMSQERR {
    static inline T eval(const T* x, int incx, const T* y, int incy) {
        T d = *x - *y;
        return d * d + SMSQERR<T, N - 1>::eval(x + incx, incx, y + incy, incy);
    }
};

template <typename T>
struct SMSQERR<T, 1> {
    static inline T eval(const T* x, int incx, const T* y, int incy) {
        T d = *x - *y;
        return d * d;
    }
};

template <typename T, int N>
struct SUMSQERR_SWITCH {
    static inline T eval(int n, const T* x, int incx, const T* y, int incy) {
        if (n == N) return SMSQERR<T, N>::eval(x, incx, y, incy);
        else return SUMSQERR_SWITCH<T, N - 1>::eval(n, x, incx, y, incy);
    }
};

template <typename T>
struct SUMSQERR_SWITCH<T, 0> {
    static inline T eval(int n, const T* x, int incx, const T* y, int incy) {
        return T();
    }
};

template <typename T, int N>
struct SUMSQERR_UNROLL {
    static inline T eval(int k, const T* x, int incx, const T* y, int incy) {
        T s = T();
        int incaN = incx * N, incbN = incy * N;
        for (int i = 0; i < k; ++i, x += incaN, y += incbN)
            s += SMSQERR<T, N>::eval(x, incx, y, incy);
        return s;
    }
};

template <typename T, int N>
struct SUMSQERR {
    static inline T eval(int n, const T* x, int incx, const T* y, int incy) {
        if (n > N)  {
            int k = n / N, kr = n % N;
            return SUMSQERR_UNROLL<T, N>::eval(k, x, incx, y, incy)
            + SUMSQERR_SWITCH<T, N>::eval(kr, x + (n - kr) * incx, incx,
                                       y + (n - kr) * incy, incy);
        } else return SUMSQERR_SWITCH<T, N>::eval(n, x, incx, y, incy);
    }
};




} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL1_IMPL_H */
