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

/** \file report.h
 * \brief Reporting.
 */

#ifndef FCNN_REPORT_H

#define FCNN_REPORT_H

#include <string>


namespace fcnn {
namespace internal {


/// Report, e.g. teaching progress (on std::cout, or somewhere else).
void report(const std::string &);


} /* namespace internal */
} /* namespace fcnn */

#endif /* FCNN_UTILS_H */
