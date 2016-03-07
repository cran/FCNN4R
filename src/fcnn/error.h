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

/** \file error.h
 *  \brief Class fcnn::exception and error handling.
 */


#ifndef FCNN_ERROR_H

#define FCNN_ERROR_H


#include <stdexcept>
#include <string>
#ifdef FCNN_DEBUG
#include <iostream>
#include <cstdlib>
#endif


namespace fcnn {

/// Exception class for handling errors in FCNN.
class exception : public std::exception {
  public:
    /// Constructor
    explicit exception(const std::string &s)
        : m_mes(s) { ; }
    /// Destructor
    virtual ~exception() throw() { ; }
    /// C string with error message
    virtual const char* what() const throw() { return m_mes.c_str(); }

  private:
    /// Error message.
    std::string m_mes;

}; /* class exception */



/// Error handling
inline
void
error(const std::string &s)
{
#ifdef FCNN_DEBUG
    std::cerr << "FCNN error: " << s << "\naborting...\n";
    abort();
#else
    throw fcnn::exception(s);
#endif
}



} /* namespace fcnn */

#endif /* FCNN_ERROR_H */
