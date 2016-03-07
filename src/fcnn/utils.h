/*
 *  This file is a part of Fast Compressed Neural Networks.
 *
 *  Copyright (c) Grzegorz Klima 2008-2016
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


/** \file utils.h
 * \brief Utility functions.
 */

#ifndef FCNN_UTILS_H

#define FCNN_UTILS_H

#include <string>
#include <vector>
#include <iostream>


namespace fcnn {
namespace internal {



/// Messages.
class message {
public:
    /// Constructor
    message() { ; }
    /// Destructor
    ~message() { ; }

    /// Conversion to std::string.
    operator std::string() const { return m_mes; }

    /// Append char.
    message& operator<<(char);
    /// Append text.
    message& operator<<(const char*);
    /// Append text.
    message& operator<<(const std::string&);
    /// Append unsigned integer.
    message& operator<<(unsigned);
    /// Append integer.
    message& operator<<(int);
    /// Append float.
    message& operator<<(float);
    /// Append double.
    message& operator<<(double);

private:
    std::string m_mes;
}; /* class message */



/// Convert number to std::string.
std::string num2str(unsigned n);
/// Convert number to std::string.
std::string num2str(int n);
/// Convert number to std::string.
std::string num2str(float n);
/// Convert number to std::string.
std::string num2str(double n);

/// Return std::string with current date and time.
std::string time_str();

/// Return std::string with FCNN version.
std::string fcnn_ver();


/// Floating point precision.
template <typename T>
struct precision {
    static const int val = 0;
};

/// Floating point precision.
template <>
struct precision<float> {
    static const int val = 7;
};

/// Floating point precision.
template <>
struct precision<double> {
    static const int val = 16;
};


/// Read (int, float or double) from input stream.
template <typename T> bool read(std::istream& is, T &n);

/// Write comment to output stream.
bool write_comment(std::ostream&, const std::string&);

/// Read comment from output stream.
bool read_comment(std::istream &is, std::string &s);

/// Skip input until the end of comment.
void skip_comment(std::istream &is);

/// Skip whitespace and newlines.
void skip_blank(std::istream &is);

/// Skip whitespace, newlines and comments.
void skip_all(std::istream &is);

/// Are we at the end of line (whitespace ignored)?
bool is_eol(std::istream &is);

/// Are we at the end of line (whitespace ignored) followed by another eol
/// or a line beginning with comment?
bool is_deol(std::istream &is);

/// Are we at the end of line or file (whitespace ignored)?
bool is_eoleof(std::istream &is);

/// Draw a random sample of size M of integers from 1 to N.
std::vector<int> sample_int(int N, int M);


} /* namespace internal */
} /* namespace fcnn */

#endif /* FCNN_UTILS_H */
