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

/** \file level1_impl.h
 *  \brief BLAS1-like routines implementation.
 */

#ifndef FCNN_LEVEL1_IMPL_H

#define FCNN_LEVEL1_IMPL_H


namespace fcnn {
namespace internal {




template <typename T, int N>
struct DOT_BLOCK {
    static inline T dot(const T* x, int incx, const T* y, int incy) {
        return *x * *y + DOT_BLOCK<T, N - 1>::dot(x + incx, incx, y + incy, incy);
    }
};


template <typename T>
struct DOT_BLOCK<T, 1> {
    static inline T dot(const T* x, int incx, const T* y, int incy) {
        return *x * *y;
    }
};

template <typename T, int N>
struct DOT_SWITCH {
    static inline T dot(int n, const T* x, int incx, const T* y, int incy) {
        if (n == N) return DOT_BLOCK<T, N>::dot(x, incx, y, incy);
        else return DOT_SWITCH<T, N - 1>::dot(n, x, incx, y, incy);
    }
};

template <typename T>
struct DOT_SWITCH<T, 0> {
    static inline T dot(int n, const T* x, int incx, const T* y, int incy) {
        return T();
    }
};

template <typename T, int N>
struct DOT_UNROLL {
    static inline T dot(int k, const T* x, int incx, const T* y, int incy) {
        T s = T();
        int incxN = incx * N, incyN = incy * N;
        for (int i = 0; i < k; ++i, x += incxN, y += incyN)
            s += DOT_BLOCK<T, N>::dot(x, incx, y, incy);
        return s;
    }
};

template <typename T, int N>
struct DOT {
    static inline T dot(int n, const T* x, int incx, const T* y, int incy) {
        if (n >= N)  {
            int k = n / N, kr = n % N;
            return DOT_UNROLL<T, N>::dot(k, x, incx, y, incy)
                   + DOT_SWITCH<T, N - 1>::dot(kr, x + (n - kr) * incx, incx, y + (n - kr) * incy, incy);
        } else return DOT_SWITCH<T, N - 1>::dot(n, x, incx, y, incy);
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
        int incxN = incx * N, incyN = incy * N;
        for (int i = 0; i < k; ++i, x += incxN, y += incyN)
            COPY_BLOCK<T, N>::copy(x, incx, y, incy);
    }
};

template <typename T, int N>
struct COPY {
    static inline void copy(int n, const T* x, int incx, T* y, int incy) {
        if (n >= N)  {
            int k = n / N, kr = n % N;
            COPY_UNROLL<T, N>::copy(k, x, incx, y, incy);
            COPY_SWITCH<T, N - 1>::copy(kr, x + (n - kr) * incx, incx, y + (n - kr) * incy, incy);
        } else COPY_SWITCH<T, N - 1>::copy(n, x, incx, y, incy);
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
        if (n >= N) {
            int k = n / N, kr = n % N;
            AXPY_UNROLL<T, N>::axpy(k, a, x, incx, y, incy);
            AXPY_SWITCH<T, N - 1>::axpy(kr, a, x + (n - kr) * incx, incx,
                                        y + (n - kr) * incy, incy);
        } else AXPY_SWITCH<T, N - 1>::axpy(n, a, x, incx, y, incy);
    }
};






template <typename T, int N>
struct DIFF_BLOCK {
    static inline void diff(const T* x, int incx, const T* y, int incy,
                            T* z, int incz) {
        *z = *x - *y;
        DIFF_BLOCK<T, N - 1>::diff(x + incx, incx, y + incy, incy, z + incz, incz);
    }
};

template <typename T>
struct DIFF_BLOCK<T, 1> {
    static inline void diff(const T* x, int incx, const T* y, int incy,
                            T* z, int incz) {
        *z = *x - *y;
    }
};

template <typename T, int N>
struct DIFF_SWITCH {
    static inline void diff(int n, const T* x, int incx, const T* y, int incy,
                            T* z, int incz) {
        if (n == N) DIFF_BLOCK<T, N>::diff(x, incx, y, incy, z, incz); else
        DIFF_SWITCH<T, N - 1>::diff(n, x, incx, y, incy, z, incz);
    }
};

template <typename T>
struct DIFF_SWITCH<T, 0> {
    static inline void diff(int n, const T* x, int incx, const T* y, int incy, T* z, int incz) {
        ;
    }
};

template <typename T, int N>
struct DIFF_UNROLL {
    static inline void diff(int k, const T* x, int incx, const T* y, int incy,
                            T* z, int incz) {
        int incxN = incx * N, incyN = incy * N, inczN = incz * N;
        for (int i = 0; i < k; ++i, x += incxN, y += incyN, z += inczN)
            DIFF_BLOCK<T, N>::diff(x, incx, y, incy, z, incz);
    }
};

template <typename T, int N>
struct DIFF {
    static inline void diff(int n, const T* x, int incx, const T* y, int incy,
                            T* z, int incz) {
        if (n >= N)  {
            int k = n / N, kr = n % N;
            DIFF_UNROLL<T, N>::diff(k, x, incx, y, incy, z, incz);
            DIFF_SWITCH<T, N - 1>::diff(kr, x + (n - kr) * incx, incx,
                                        y + (n - kr) * incy, incy,
                                        z + (n - kr) * incz, incz);
        } else DIFF_SWITCH<T, N - 1>::diff(n, x, incx, y, incy, z, incz);
    }
};







template <typename T, int N>
struct SMSQDIFF {
    static inline T sumsqdiff(const T* x, int incx, const T* y, int incy) {
        T d = *x - *y;
        return d * d + SMSQDIFF<T, N - 1>::sumsqdiff(x + incx, incx, y + incy, incy);
    }
};

template <typename T>
struct SMSQDIFF<T, 1> {
    static inline T sumsqdiff(const T* x, int incx, const T* y, int incy) {
        T d = *x - *y;
        return d * d;
    }
};

template <typename T, int N>
struct SUMSQDIFF_SWITCH {
    static inline T sumsqdiff(int n, const T* x, int incx, const T* y, int incy) {
        if (n == N) return SMSQDIFF<T, N>::sumsqdiff(x, incx, y, incy);
        else return SUMSQDIFF_SWITCH<T, N - 1>::sumsqdiff(n, x, incx, y, incy);
    }
};

template <typename T>
struct SUMSQDIFF_SWITCH<T, 0> {
    static inline T sumsqdiff(int n, const T* x, int incx, const T* y, int incy) {
        return T();
    }
};

template <typename T, int N>
struct SUMSQDIFF_UNROLL {
    static inline T sumsqdiff(int k, const T* x, int incx, const T* y, int incy) {
        T s = T();
        int incxN = incx * N, incyN = incy * N;
        for (int i = 0; i < k; ++i, x += incxN, y += incyN)
            s += SMSQDIFF<T, N>::sumsqdiff(x, incx, y, incy);
        return s;
    }
};

template <typename T, int N>
struct SUMSQDIFF {
    static inline T sumsqdiff(int n, const T* x, int incx, const T* y, int incy) {
        if (n >= N)  {
            int k = n / N, kr = n % N;
            return SUMSQDIFF_UNROLL<T, N>::sumsqdiff(k, x, incx, y, incy)
            + SUMSQDIFF_SWITCH<T, N - 1>::sumsqdiff(kr, x + (n - kr) * incx, incx,
                                       y + (n - kr) * incy, incy);
        } else return SUMSQDIFF_SWITCH<T, N - 1>::sumsqdiff(n, x, incx, y, incy);
    }
};




template <typename T, int N>
struct SUMSQ_BLOCK {
    static inline T sumsq(const T* x, int incx) {
        return *x * *x + SUMSQ_BLOCK<T, N - 1>::sumsq(x + incx, incx);
    }
};


template <typename T>
struct SUMSQ_BLOCK<T, 1> {
    static inline T sumsq(const T* x, int incx) {
        return *x * *x;
    }
};

template <typename T, int N>
struct SUMSQ_SWITCH {
    static inline T sumsq(int n, const T* x, int incx) {
        if (n == N) return SUMSQ_BLOCK<T, N>::sumsq(x, incx);
        else return SUMSQ_SWITCH<T, N - 1>::sumsq(n, x, incx);
    }
};

template <typename T>
struct SUMSQ_SWITCH<T, 0> {
    static inline T sumsq(int n, const T* x, int incx) {
        return T();
    }
};

template <typename T, int N>
struct SUMSQ_UNROLL {
    static inline T sumsq(int k, const T* x, int incx) {
        T s = T();
        int incxN = incx * N;
        for (int i = 0; i < k; ++i, x += incxN)
            s += SUMSQ_BLOCK<T, N>::sumsq(x, incx);
        return s;
    }
};

template <typename T, int N>
struct SUMSQ {
    static inline T sumsq(int n, const T* x, int incx) {
        if (n >= N)  {
            int k = n / N, kr = n % N;
            return SUMSQ_UNROLL<T, N>::sumsq(k, x, incx)
                   + SUMSQ_SWITCH<T, N - 1>::sumsq(kr, x + (n - kr) * incx, incx);
        } else return SUMSQ_SWITCH<T, N - 1>::sumsq(n, x, incx);
    }
};





} /* namespace internal */
} /* namespace fcnn */


#endif /* FCNN_LEVEL1_IMPL_H */
