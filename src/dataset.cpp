/*
 *  This file is a part of FCNN4R.
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

/** \file dataset.cpp
 *  \brief Reading and writing dataset in the FCNN format.
 */


#include <fcnn/utils.h>
#include <Rcpp.h>
#include <fstream>

using namespace fcnn::internal;


RcppExport
SEXP
read_fcnndataset(SEXP fname)
{
    std::string fn = Rcpp::as<std::string>(fname);
    int r, ci, co;
    std::vector<std::string> rinfo;
    std::vector<double> inp;
    std::vector<double> outp;

    std::ifstream is;
    is.open(fn.c_str());
    if (is.fail()) goto err;
    skip_blank(is);
    skip_comment(is);
    if (is.fail()) goto err;
    if (is_eol(is)) goto err;
    if (!read<int>(is, r)) goto err;
    if (is_eol(is)) goto err;
    if (!read<int>(is, ci)) goto err;
    if (is_eol(is)) goto err;
    if (!read<int>(is, co)) goto err;
    if (!is_eol(is)) goto err;
    skip_blank(is);
    if (is.fail()) goto err;
    if ((r < 1) || (ci < 1) || (co < 1)) goto err;

    rinfo.reserve(r);
    inp.reserve(r * ci);
    outp.reserve(r * co);

    for  (int i = 1; i <= r; ++i) {
        std::string cm;
        double d;
        if (read_comment(is, cm)) rinfo.push_back(cm);
        else rinfo.push_back("");
        for (int j = 1; j <= ci; ++j) {
            if (is_eol(is)) goto err;
            if (!read<double>(is, d)) goto err;
            inp.push_back(d);
        }
        if (!is_eol(is)) goto err;
        for (int j = 1; j <= co; ++j) {
            if (is_eol(is)) goto err;
            if (!read<double>(is, d)) goto err;
            outp.push_back(d);
        }
        if (i < r) {
            if (!is_eol(is)) goto err;
        } else {
            skip_all(is);
            if (!is.eof()) goto err;
        }
    }

    SEXP wrap_r, wrap_ci, wrap_co, wrap_rinfo, wrap_inp, wrap_outp,
         ret;
    PROTECT(wrap_r = Rcpp::wrap(r));
    PROTECT(wrap_ci = Rcpp::wrap(ci));
    PROTECT(wrap_co = Rcpp::wrap(co));
    PROTECT(wrap_rinfo = Rcpp::wrap(rinfo));
    PROTECT(wrap_inp = Rcpp::wrap(inp));
    PROTECT(wrap_outp = Rcpp::wrap(outp));
    PROTECT(ret = Rcpp::List::create(wrap_r,
                                     wrap_ci,
                                     wrap_co,
                                     wrap_rinfo,
                                     wrap_inp,
                                     wrap_outp));
    UNPROTECT(7);
    return ret;
err:
    return R_NilValue;
}


RcppExport
SEXP
write_fcnndataset(SEXP fname,
                  SEXP norec, SEXP noinp, SEXP nooutp,
                  SEXP rowinfo, SEXP inp, SEXP outp)
{
    std::string fn = Rcpp::as<std::string>(fname);
    int r = Rcpp::as<int>(norec);
    int ci = Rcpp::as<int>(noinp);
    int co = Rcpp::as<int>(nooutp);
    std::vector<std::string> rinfo = Rcpp::as<std::vector<std::string> >(rowinfo);
    double *in = REAL(inp), *out = REAL(outp);
    std::ofstream os;
    os.open(fn.c_str());
    if (os.fail()) return Rcpp::wrap(false);
    int i, j;

    if (!write_comment(os, "saved " + time_str())) return Rcpp::wrap(false);
    os << r << ' ' << ci << ' ' << co << "\n\n";
    os << std::setprecision(precision<double>::val);
    for (i = 0; i < r; ++i) {
        if (rinfo[i].empty()) {
            if (!write_comment(os, num2str(i + 1)))
                return Rcpp::wrap(false);
        } else {
            if (!write_comment(os, rinfo[i])) return Rcpp::wrap(false);
        }
        for (j = 1; j < ci; ++j) os << in[(j - 1) * r + i] << ' ';
        os << in[(j - 1) * r + i] << '\n';
        if (os.fail()) return Rcpp::wrap(false);
        for (j = 1; j < co; ++j) os << out[(j - 1) * r + i] << ' ';
        os << out[(j - 1) * r + i] << '\n';
        if (os.fail()) return Rcpp::wrap(false);
    }
    os << '\n';

    os.close();
    if (os.fail()) return Rcpp::wrap(false);
    return Rcpp::wrap(true);
}











