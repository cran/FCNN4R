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

/** \file utils.cpp
 * \brief Utility functions.
 */

#include <fcnn/utils.h>
#include <fcnn/fcnncfg.h>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <ctime>



using namespace std;
using namespace fcnn;
using namespace fcnn::internal;



message&
message::operator<<(char c)
{
    m_mes += c;
    return *this;
}


message&
message::operator<<(const char *s)
{
    m_mes += s;
    return *this;
}


message&
message::operator<<(const string &s)
{
    m_mes += s;
    return *this;
}


message&
message::operator<<(unsigned n)
{
    m_mes += num2str(n);
    return *this;
}


message&
message::operator<<(int n)
{
    m_mes += num2str(n);
    return *this;
}



message&
message::operator<<(float n)
{
    m_mes += num2str(n);
    return *this;
}



message&
message::operator<<(double n)
{
    m_mes += num2str(n);
    return *this;
}



string
fcnn::internal::num2str(double n)
{
    char buf[30];
    sprintf(buf, "%g", n);
    return string(buf);
}


string
fcnn::internal::num2str(float n)
{
    char buf[30];
    sprintf(buf, "%g", n);
    return string(buf);
}


string
fcnn::internal::num2str(int n)
{
    char buf[15];
    sprintf(buf, "%d", n);
    return string(buf);
}


string
fcnn::internal::num2str(unsigned n)
{
    char buf[15];
    sprintf(buf, "%u", n);
    return string(buf);
}


string
fcnn::internal::time_str()
{
    string ts, mo, da, ho, mi, se;
    time_t tt = time(0);
    struct tm *now = localtime(&tt);
    mo = (now->tm_mon < 9) ? '0' + num2str(now->tm_mon + 1) : num2str(now->tm_mon + 1);
    da = (now->tm_mday < 10) ? '0' + num2str(now->tm_mday) : num2str(now->tm_mday);
    ho = (now->tm_hour < 10) ? '0' + num2str(now->tm_hour) : num2str(now->tm_hour);
    mi = (now->tm_min < 10) ? '0' + num2str(now->tm_min) : num2str(now->tm_min);
    se = (now->tm_sec < 10) ? '0' + num2str(now->tm_sec) : num2str(now->tm_sec);
    ts = num2str(now->tm_year + 1900) + '-' + mo + '-' + da + ' '
         + ho + ':' + mi + ':' + se;
    return ts;
}


string
fcnn::internal::fcnn_ver()
{
    return string(VERSION);
}



template <typename T>
bool
fcnn::internal::read(istream &is, T &n)
{
    char c;
    T x;

    is >> x;
    if (is.fail() || is.bad()) return false;
    else
    {
        if (is.eof()) {
            n = x;
            return true;
        }
        c = is.peek();
        if ((c == ' ') || (c == '\t') || (c == '\n')) {
            n = x;
            return true;
        }
    }
    return false;
}


// Instantiations
template bool fcnn::internal::read<unsigned>(istream &s, unsigned &n);
template bool fcnn::internal::read<int>(istream &s, int &n);
template bool fcnn::internal::read<float>(istream &s, float &n);
template bool fcnn::internal::read<double>(istream &s, double &n);


bool
fcnn::internal::write_comment(ostream &os, const string &s)
{
    const char *c = s.c_str();
    if (*c) os << "# "; else return true;
    while (*c) {
        os << *c;
        if ((*c == '\n') && (c[1])) os << "# ";
        ++c;
    }
    os << '\n';
    if (os.fail()) return false;
    return true;
}



bool
fcnn::internal::read_comment(istream &is, string &s)
{
    if (!is.good() || is.eof()) return false;
    char c;
    s.clear();
    c = is.peek();
    if (c != '#') return false;
    is.get(c);

start_line:
    do { is.get(c); }
    while ((is.good() && !is.eof()) && ((c == ' ') || (c == '\t')));
    if (is.eof()) return true;
    if (is.fail()) return false;
    while ((is) && (c != '\n')) {
        s += c;
        is.get(c);
    }
    if (is.eof()) return true;
    if (is.fail()) return false;
    if (c == '\n') {
        if (is.peek() == '#') { s += c; goto start_line; }
        else return true;
    }
    return true;
}


void
fcnn::internal::skip_comment(istream &is)
{
    if (!is.good() || is.eof()) return;
    char c;
    c = is.peek();
    if (c != '#') return;
    is.get(c);

start_line:
    do { is.get(c); } while ((is.good() && !is.eof()) && (c != '\n'));
    if (!is) return;
    if (c == '\n') {
        if (is.peek() == '#') goto start_line;
        else return;
    }
}


void
fcnn::internal::skip_blank(istream &is)
{
    char c;
    while (is.good() && !is.eof()) {
        is.get(c);
        if ((c != ' ') && (c != '\t') && (c != '\n') && (c != '\0')) {
            is.putback(c);
            break;
        }
    }
}


void
fcnn::internal::skip_all(istream &is)
{
    char c;
    while (is.good() && !is.eof()) {
        is.get(c);
        if ((c != ' ') && (c != '\t') && (c != '\n') && (c != '\0')) {
            is.putback(c);
            if (c == '#') {
                skip_comment(is);
            } else {
                break;
            }
        }
    }
}



bool
fcnn::internal::is_eol(istream &is)
{
    char c;
    while (is.good() && !is.eof()) {
        is.get(c);
        if ((c != ' ') && (c != '\t')) {
            if (c == '\n') return true;
            else {
                is.putback(c);
                break;
            }
        }
    }
    return false;
}



bool
fcnn::internal::is_deol(istream &is)
{
    if (!is_eol(is)) return false;
    char c;
    c = is.peek();
    if (c == '#') return true;
    return is_eol(is);
}



bool
fcnn::internal::is_eoleof(istream &is)
{
    char c;
    if (is.eof()) return true;
    while (is) {
        if (is.eof()) return true;
        is.get(c);
        if ((c != ' ') && (c != '\t')) {
            if ((c == '\n') || (c == '\0')) return true;
            else {
                is.putback(c);
                return false;
            }
        }
    }
    return false;
}



// Not needed under R...
#ifndef R_SHAREDLIB

std::vector<int>
fcnn::internal::sample_int(int N, int M)
{
    std::vector<int> res;
    std::set<int> help;

    if ((N < M) || (M < 1)) return res;

    int i;
    res.reserve(M);
    while (help.size() < M) {
        i = (int)((double) ::rand() / ((double) RAND_MAX + 1.) * (double) N) + 1;
        if (help.insert(i).second) res.push_back(i);
    }
    return res;
}

#endif /* R_SHAREDLIB */


