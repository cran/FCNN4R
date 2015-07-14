/*
 *  This file is a part of FCNN4R.
 *
 *  Copyright (c) Grzegorz Klima 2015
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

/** \file interface.cpp
 *  \brief R interface to functions working with multilayer perceptron network structure.
 */


#include <fcnn/struct.h>
#include <fcnn/export.h>
#include <fcnn/level3.h>
#include <fcnn/activation.h>
#include <Rcpp.h>


RcppExport
SEXP
mlp_construct(SEXP lays)
{
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p;
    std::vector<int> n_prev;
    std::vector<int> n_next;
    std::vector<int> w_p;
    std::vector<double> w_val;
    std::vector<int> w_fl;
    int w_on;

    try {
        fcnn::internal::mlp_construct(layers, n_p, n_prev, n_next,
                                    w_p, w_val, w_fl, w_on);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }

    return Rcpp::List::create(Rcpp::wrap(n_p),
                              Rcpp::wrap(n_prev),
                              Rcpp::wrap(n_next),
                              Rcpp::wrap(w_p),
                              Rcpp::wrap(w_val),
                              Rcpp::wrap(w_fl),
                              Rcpp::wrap(w_on));
}


RcppExport
SEXP
mlp_rm_neurons(SEXP lays, SEXP np, SEXP nprev, SEXP nnext,
               SEXP wp, SEXP wval, SEXP wfl, SEXP won,
               SEXP hlaf, SEXP hlafp, SEXP reprt)
{
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<int> n_prev = Rcpp::as<std::vector<int> >(nprev);
    std::vector<int> n_next = Rcpp::as<std::vector<int> >(nnext);
    std::vector<int> w_p = Rcpp::as<std::vector<int> >(wp);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    int w_on = Rcpp::as<int>(won);
    int hl_af = Rcpp::as<int>(hlaf);
    double hl_af_p = Rcpp::as<double>(hlafp);
    bool report = Rcpp::as<bool>(reprt);

    int nrm = fcnn::internal::mlp_rm_neurons(layers, n_p, n_prev, n_next,
                                             w_p, w_val, w_fl, w_on,
                                             hl_af, hl_af_p,
                                             report);

    return Rcpp::List::create(Rcpp::wrap(layers),
                              Rcpp::wrap(n_p),
                              Rcpp::wrap(n_prev),
                              Rcpp::wrap(n_next),
                              Rcpp::wrap(w_p),
                              Rcpp::wrap(w_val),
                              Rcpp::wrap(w_fl),
                              Rcpp::wrap(w_on),
                              Rcpp::wrap(nrm));
}


RcppExport
SEXP
mlp_import(SEXP fname)
{
    std::string fn = Rcpp::as<std::string>(fname);
    std::string netname;
    std::vector<int> layers;
    std::vector<int> n_p;
    std::vector<int> n_prev;
    std::vector<int> n_next;
    std::vector<int> w_p;
    std::vector<double> w_val;
    std::vector<int> w_fl;
    int w_on;
    int hl_af;
    double hl_af_p;
    int ol_af;
    double ol_af_p;

    if (!fcnn::internal::mlp_load_txt(fn,
                                      netname,
                                      layers, n_p, n_prev, n_next,
                                      w_p, w_val, w_fl, w_on,
                                      hl_af, hl_af_p, ol_af, ol_af_p)) {
        return R_NilValue;
    }

    return Rcpp::List::create(Rcpp::wrap(netname),
                              Rcpp::wrap(layers),
                              Rcpp::wrap(n_p),
                              Rcpp::wrap(n_prev),
                              Rcpp::wrap(n_next),
                              Rcpp::wrap(w_p),
                              Rcpp::wrap(w_val),
                              Rcpp::wrap(w_fl),
                              Rcpp::wrap(w_on),
                              Rcpp::wrap(hl_af),
                              Rcpp::wrap(hl_af_p),
                              Rcpp::wrap(ol_af),
                              Rcpp::wrap(ol_af_p));
}


RcppExport
SEXP
mlp_export(SEXP fname,
           SEXP nname, SEXP lays, SEXP wval, SEXP wfl,
           SEXP hlaf, SEXP hlafp, SEXP olaf, SEXP olafp)
{
    std::string fn = Rcpp::as<std::string>(fname);
    std::string netname = Rcpp::as<std::string>(nname);
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    int hl_af = Rcpp::as<int>(hlaf);
    double hl_af_p = Rcpp::as<double>(hlafp);
    int ol_af = Rcpp::as<int>(olaf);
    double ol_af_p = Rcpp::as<double>(olafp);

    return Rcpp::wrap(fcnn::internal::mlp_save_txt(fn,
                                                   netname, layers, w_val, w_fl,
                                                   hl_af, hl_af_p, ol_af, ol_af_p));
}


RcppExport
SEXP
mlp_export_C(SEXP fname,
             SEXP nname, SEXP lays, SEXP np, SEXP wval,
             SEXP hlaf, SEXP hlafp, SEXP olaf, SEXP olafp,
             SEXP A, SEXP b, SEXP C, SEXP d)
{
    std::string fn = Rcpp::as<std::string>(fname);
    std::string netname = Rcpp::as<std::string>(nname);
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    int hl_af = Rcpp::as<int>(hlaf);
    double hl_af_p = Rcpp::as<double>(hlafp);
    int ol_af = Rcpp::as<int>(olaf);
    double ol_af_p = Rcpp::as<double>(olafp);

    return Rcpp::wrap(fcnn::internal::mlp_export_C(fn,
                                                   netname, layers, n_p, w_val,
                                                   hl_af, hl_af_p, ol_af, ol_af_p,
                                                   REAL(A), REAL(b), REAL(C), REAL(d)));
}



RcppExport
SEXP
mlp_get_abs_w_idx(SEXP wfl, SEXP idx)
{
    const int *w_fl = INTEGER(wfl);
    int i = Rcpp::as<int>(idx);
    int res = fcnn::internal::mlp_get_abs_w_idx(w_fl, i);
    return Rcpp::wrap(res);
}



RcppExport
void
mlp_set_active(const int *layers, const int *n_p, int *n_prev, int *n_next,
               const int *w_p, double *w_val, int *w_fl, int *w_on,
               int *i, int *on)
{
    fcnn::internal::mlp_set_active(layers, n_p, n_prev, n_next,
                                   w_p, w_val, w_fl, w_on,
                                   *i, *on);
}




RcppExport
SEXP
actvfuncstr(SEXP idx)
{
    return Rcpp::wrap(fcnn::internal::mlp_act_f_str(Rcpp::as<int>(idx)));
}


RcppExport
void
mlp_eval(const int *lays, const int *no_lays, const int *n_pts,
         const double *w_val, const int *hl_af, const double *hl_af_p,
         const int *ol_af, const double *ol_af_p,
         const int *no_datarows, const double *in, double *out)
{
    fcnn::internal::eval(lays, *no_lays, n_pts,
                         w_val, *hl_af, *hl_af_p, *ol_af, *ol_af_p,
                         *no_datarows, in, out);
}



RcppExport
void
mlp_mse(const int *lays, const int *no_lays, const int *n_pts,
        const double *w_val, const int *hl_af, const double *hl_af_p,
        const int *ol_af, const double *ol_af_p,
        const int *no_datarows, const double *in, const double *out,
        double *res)
{
    *res = fcnn::internal::mse(lays, *no_lays, n_pts,
                               w_val, *hl_af, *hl_af_p, *ol_af, *ol_af_p,
                               *no_datarows, in, out);
}



RcppExport
void
mlp_grad(const int *lays, const int *no_lays, const int *n_pts,
         const int *w_pts, const int *w_fl, const double *w_val,
         const int *hl_af, const double *hl_af_p,
         const int *ol_af, const double *ol_af_p,
         const int *no_datarows, const double *in, const double *out,
         double *msegrad)
{
    double mse = fcnn::internal::grad(lays, *no_lays, n_pts,
                                      w_pts, w_fl, w_val,
                                      *hl_af, *hl_af_p, *ol_af, *ol_af_p,
                                      *no_datarows, in, out, msegrad + 1);
    msegrad[0] = mse;
}



RcppExport
void
mlp_gradi(const int *lays, const int *no_lays, const int *n_pts,
          const int *w_pts, const int *w_fl, const double *w_val,
          const int *hl_af, const double *hl_af_p,
          const int *ol_af, const double *ol_af_p,
          const int *no_datarows, const int *i, double *in, const double *out,
          double *grad)
{
    fcnn::internal::gradi(lays, *no_lays, n_pts,
                          w_pts, w_fl, w_val,
                          *hl_af, *hl_af_p, *ol_af, *ol_af_p,
                          *no_datarows, *i - 1, in, out, grad);
}



RcppExport
void
mlp_gradij(const int *lays, const int *no_lays, const int *n_pts,
           const int *w_pts, const int *w_fl, const double *w_val, const int *no_w_on,
           const int *hl_af, const double *hl_af_p,
           const int *ol_af, const double *ol_af_p,
           const int *no_datarows, const int *i, double *in,
           double *grad)
{
    fcnn::internal::gradij(lays, *no_lays, n_pts,
                           w_pts, w_fl, w_val, *no_w_on,
                           *hl_af, *hl_af_p, *ol_af, *ol_af_p,
                           *no_datarows, *i - 1, in, grad);
}
















