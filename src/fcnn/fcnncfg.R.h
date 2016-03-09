/*
 *  This file is a part of FCNN4R.
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

/** \file fcnncfg.R.h
 *  \brief FCNN R configuration file.
 */

#ifndef FCNNCFG_R_H

#define FCNNCFG_R_H

#undef F77_DUMMY_MAIN
#undef F77_FUNC
#undef F77_FUNC_
#undef FC_DUMMY_MAIN_EQ_F77
#undef HAVE_BLAS
#undef HAVE_DLFCN_H
#undef HAVE_GETTIMEOFDAY
#undef HAVE_INTTYPES_H
#undef HAVE_LAPACK
#undef HAVE_LIBM
#undef HAVE_MEMORY_H
#undef HAVE_OPENMP
#undef HAVE_STDINT_H
#undef HAVE_STDLIB_H
#undef HAVE_STRINGS_H
#undef HAVE_STRING_H
#undef HAVE_SYS_STAT_H
#undef HAVE_SYS_TIME_H
#undef HAVE_SYS_TYPES_H
#undef HAVE_UNISTD_H
#undef LT_OBJDIR
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION
#undef STDC_HEADERS

#define HAVE_BLAS 1
#include <R_ext/RS.h>
#define F77_FUNC(name,NAME) F77_NAME(name)

#include <Rconfig.h>
#if defined(_OPENMP) || defined(SUPPORT_OPENMP)
#define HAVE_OPENMP 1
#endif /* defined(_OPENMP) || defined(SUPPORT_OPENMP) */

#endif /* FCNNCFG_R_H */
