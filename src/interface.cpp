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
#include <fcnn/utils.h>
#include <Rcpp.h>


RcppExport
SEXP
fcnn_ver()
{
    return Rcpp::wrap(fcnn::internal::fcnn_ver());
}


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

    SEXP wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl, wrap_w_on,
         ret;
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(wrap_w_on = Rcpp::wrap(w_on));
    PROTECT(ret = Rcpp::List::create(wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl,
                                     wrap_w_on));
    UNPROTECT(8);
    return ret;
}





RcppExport
SEXP
mlp_expand_reorder_inputs(SEXP lays, SEXP np, SEXP nprev, SEXP nnext,
                          SEXP wp, SEXP wval, SEXP wfl,
                          SEXP nn, SEXP m)
{
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<int> n_prev = Rcpp::as<std::vector<int> >(nprev);
    std::vector<int> n_next = Rcpp::as<std::vector<int> >(nnext);
    std::vector<int> w_p = Rcpp::as<std::vector<int> >(wp);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    int nnew = Rcpp::as<int>(nn);
    std::vector<int> mi = Rcpp::as<std::vector<int> >(m);
    std::map<int, int> mapi;
    for (int i = 0; i < mi.size(); ++i) mapi[i + 1] = mi[i];

    try {
        fcnn::internal::mlp_expand_reorder_inputs(layers, n_p, n_prev, n_next,
                                                  w_p, w_val, w_fl, nnew, mapi);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }

    SEXP wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl,
         ret;
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(ret = Rcpp::List::create(wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl));
    UNPROTECT(8);
    return ret;
}


RcppExport
SEXP
mlp_rm_neurons(SEXP lays, SEXP np, SEXP nprev, SEXP nnext,
               SEXP wp, SEXP wval, SEXP wfl, SEXP won,
               SEXP a, SEXP ap, SEXP reprt)
{
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<int> n_prev = Rcpp::as<std::vector<int> >(nprev);
    std::vector<int> n_next = Rcpp::as<std::vector<int> >(nnext);
    std::vector<int> w_p = Rcpp::as<std::vector<int> >(wp);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    int w_on = Rcpp::as<int>(won);
    std::vector<int> af = Rcpp::as<std::vector<int> >(a);
    std::vector<double> af_p = Rcpp::as<std::vector<double> >(ap);
    bool report = Rcpp::as<bool>(reprt);

    int nrm = fcnn::internal::mlp_rm_neurons(layers, n_p, n_prev, n_next,
                                             w_p, w_val, w_fl, w_on,
                                             af, af_p,
                                             report);

    SEXP wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl, wrap_w_on,
         wrap_nrm,
         ret;
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(wrap_w_on = Rcpp::wrap(w_on));
    PROTECT(wrap_nrm = Rcpp::wrap(nrm));
    PROTECT(ret = Rcpp::List::create(wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl,
                                     wrap_w_on,
                                     wrap_nrm));
    UNPROTECT(10);
    return ret;
}




RcppExport
SEXP
mlp_rm_input_neurons(SEXP lays, SEXP np, SEXP nprev, SEXP nnext,
                     SEXP wp, SEXP wval, SEXP wfl, SEXP reprt)
{
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<int> n_prev = Rcpp::as<std::vector<int> >(nprev);
    std::vector<int> n_next = Rcpp::as<std::vector<int> >(nnext);
    std::vector<int> w_p = Rcpp::as<std::vector<int> >(wp);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    bool report = Rcpp::as<bool>(reprt);

    fcnn::internal::mlp_rm_input_neurons(layers, n_p, n_prev, n_next,
                                         w_p, w_val, w_fl, report);

    SEXP wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl,
         ret;
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(ret = Rcpp::List::create(wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl));
    UNPROTECT(8);
    return ret;
}



RcppExport
SEXP
mlp_merge(SEXP Alays, SEXP Awp, SEXP Awval, SEXP Awfl,
          SEXP Blays, SEXP Bwp, SEXP Bwval, SEXP Bwfl,
          SEXP sinp)
{
    std::vector<int> Alayers = Rcpp::as<std::vector<int> >(Alays);
    std::vector<int> Aw_p = Rcpp::as<std::vector<int> >(Awp);
    std::vector<double> Aw_val = Rcpp::as<std::vector<double> >(Awval);
    std::vector<int> Aw_fl = Rcpp::as<std::vector<int> >(Awfl);
    std::vector<int> Blayers = Rcpp::as<std::vector<int> >(Blays);
    std::vector<int> Bw_p = Rcpp::as<std::vector<int> >(Bwp);
    std::vector<double> Bw_val = Rcpp::as<std::vector<double> >(Bwval);
    std::vector<int> Bw_fl = Rcpp::as<std::vector<int> >(Bwfl);
    bool same_inputs = Rcpp::as<bool>(sinp);
    std::vector<int> layers;
    std::vector<int> n_p;
    std::vector<int> n_prev;
    std::vector<int> n_next;
    std::vector<int> w_p;
    std::vector<double> w_val;
    std::vector<int> w_fl;
    int w_on;

    try {
        fcnn::internal::mlp_merge(Alayers, Aw_p, Aw_val, Aw_fl,
                                  Blayers, Bw_p, Bw_val, Bw_fl,
                                  same_inputs,
                                  layers, n_p, n_prev, n_next,
                                  w_p, w_val, w_fl, w_on);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }

    SEXP wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl, wrap_w_on,
         ret;
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(wrap_w_on = Rcpp::wrap(w_on));
    PROTECT(ret = Rcpp::List::create(wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl,
                                     wrap_w_on));
    UNPROTECT(9);
    return ret;
}


RcppExport
SEXP
mlp_stack(SEXP Alays, SEXP Awp, SEXP Awval, SEXP Awfl,
          SEXP Blays, SEXP Bwp, SEXP Bwval, SEXP Bwfl)
{
    std::vector<int> Alayers = Rcpp::as<std::vector<int> >(Alays);
    std::vector<int> Aw_p = Rcpp::as<std::vector<int> >(Awp);
    std::vector<double> Aw_val = Rcpp::as<std::vector<double> >(Awval);
    std::vector<int> Aw_fl = Rcpp::as<std::vector<int> >(Awfl);
    std::vector<int> Blayers = Rcpp::as<std::vector<int> >(Blays);
    std::vector<int> Bw_p = Rcpp::as<std::vector<int> >(Bwp);
    std::vector<double> Bw_val = Rcpp::as<std::vector<double> >(Bwval);
    std::vector<int> Bw_fl = Rcpp::as<std::vector<int> >(Bwfl);
    std::vector<int> layers;
    std::vector<int> n_p;
    std::vector<int> n_prev;
    std::vector<int> n_next;
    std::vector<int> w_p;
    std::vector<double> w_val;
    std::vector<int> w_fl;
    int w_on;

    try {
        fcnn::internal::mlp_stack(Alayers, Aw_p, Aw_val, Aw_fl,
                                  Blayers, Bw_p, Bw_val, Bw_fl,
                                  layers, n_p, n_prev, n_next,
                                  w_p, w_val, w_fl, w_on);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }

    SEXP wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl, wrap_w_on,
         ret;
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(wrap_w_on = Rcpp::wrap(w_on));
    PROTECT(ret = Rcpp::List::create(wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl,
                                     wrap_w_on));
    UNPROTECT(9);
    return ret;
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
    std::vector<int> af;
    std::vector<double> af_p;

    if (!fcnn::internal::mlp_load_txt(fn,
                                      netname,
                                      layers, n_p, n_prev, n_next,
                                      w_p, w_val, w_fl, w_on,
                                      af, af_p)) {
        return R_NilValue;
    }
    SEXP wrap_netname, wrap_layers, wrap_n_p, wrap_n_prev, wrap_n_next,
         wrap_w_p, wrap_w_val, wrap_w_fl, wrap_w_on,
         wrap_af, wrap_af_p,
         ret;
    PROTECT(wrap_netname = Rcpp::wrap(netname));
    PROTECT(wrap_layers = Rcpp::wrap(layers));
    PROTECT(wrap_n_p = Rcpp::wrap(n_p));
    PROTECT(wrap_n_prev = Rcpp::wrap(n_prev));
    PROTECT(wrap_n_next = Rcpp::wrap(n_next));
    PROTECT(wrap_w_p = Rcpp::wrap(w_p));
    PROTECT(wrap_w_val = Rcpp::wrap(w_val));
    PROTECT(wrap_w_fl = Rcpp::wrap(w_fl));
    PROTECT(wrap_w_on = Rcpp::wrap(w_on));
    PROTECT(wrap_af = Rcpp::wrap(af));
    PROTECT(wrap_af_p = Rcpp::wrap(af_p));
    PROTECT(ret = Rcpp::List::create(wrap_netname,
                                     wrap_layers,
                                     wrap_n_p,
                                     wrap_n_prev,
                                     wrap_n_next,
                                     wrap_w_p,
                                     wrap_w_val,
                                     wrap_w_fl,
                                     wrap_w_on,
                                     wrap_af,
                                     wrap_af_p));
    UNPROTECT(12);
    return ret;
}


RcppExport
SEXP
mlp_export(SEXP fname,
           SEXP nname, SEXP lays, SEXP wval, SEXP wfl,
           SEXP a, SEXP ap)
{
    std::string fn = Rcpp::as<std::string>(fname);
    std::string netname = Rcpp::as<std::string>(nname);
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    std::vector<int> af = Rcpp::as<std::vector<int> >(a);
    std::vector<double> af_p = Rcpp::as<std::vector<double> >(ap);

    return Rcpp::wrap(fcnn::internal::mlp_save_txt(fn,
                                                   netname, layers, w_val, w_fl,
                                                   af, af_p));
}


RcppExport
SEXP
mlp_export_C(SEXP fname,
             SEXP nname, SEXP lays, SEXP np, SEXP wval, SEXP wfl, SEXP won,
             SEXP a, SEXP ap, SEXP bp,
             SEXP A, SEXP b, SEXP C, SEXP d, SEXP E, SEXP f)
{
    std::string fn = Rcpp::as<std::string>(fname);
    std::string netname = Rcpp::as<std::string>(nname);
    std::vector<int> layers = Rcpp::as<std::vector<int> >(lays);
    std::vector<int> n_p = Rcpp::as<std::vector<int> >(np);
    std::vector<double> w_val = Rcpp::as<std::vector<double> >(wval);
    std::vector<int> w_fl = Rcpp::as<std::vector<int> >(wfl);
    int w_on = Rcpp::as<int>(won);
    std::vector<int> af = Rcpp::as<std::vector<int> >(a);
    std::vector<double> af_p = Rcpp::as<std::vector<double> >(ap);
    bool with_bp = Rcpp::as<bool>(bp);
    double *Aptr = Rf_isNull(A) ? 0 : REAL(A), *bptr = Rf_isNull(b) ? 0 : REAL(b),
           *Cptr = Rf_isNull(C) ? 0 : REAL(C), *dptr = Rf_isNull(d) ? 0 : REAL(d),
           *Eptr = Rf_isNull(E) ? 0 : REAL(E), *fptr = Rf_isNull(f) ? 0 : REAL(f);

    try {
        return Rcpp::wrap(fcnn::internal::mlp_export_C(fn,
                                                       netname, layers, n_p, w_val, w_fl, w_on,
                                                       af, af_p, with_bp,
                                                       Aptr, bptr, Cptr, dptr, Eptr, fptr));
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }
}



RcppExport
SEXP
mlp_get_abs_w_idx(SEXP wfl, SEXP idx)
{
    const int *w_fl = INTEGER(wfl);
    std::vector<int> ii = Rcpp::as<std::vector<int> >(idx);
    std::vector<int> ai(ii.size());
    for (int i = 0; i < ii.size(); ++i) {
        ai[i] = fcnn::internal::mlp_get_abs_w_idx(w_fl, ii[i]);
    }
    return Rcpp::wrap(ai);
}



RcppExport
void
mlp_set_active(const int *layers, const int *n_p, int *n_prev, int *n_next,
               const int *w_p, double *w_val, int *w_fl, int *w_on,
               int *i, int *on, int *N)
{
    for (int n = 0; n < *N; ++n) {
        fcnn::internal::mlp_set_active(layers, n_p, n_prev, n_next,
                                    w_p, w_val, w_fl, w_on,
                                    i[n], on[n]);
    }
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
         const double *w_val, const int *af, const double *af_p,
         const int *no_datarows, const double *in, double *out)
{
    fcnn::internal::eval(lays, *no_lays, n_pts,
                         w_val, af, af_p,
                         *no_datarows, in, out);
}



RcppExport
void
mlp_mse(const int *lays, const int *no_lays, const int *n_pts,
        const double *w_val, const int *af, const double *af_p,
        const int *no_datarows, const double *in, const double *out,
        double *res)
{
    *res = fcnn::internal::mse(lays, *no_lays, n_pts,
                               w_val, af, af_p,
                               *no_datarows, in, out);
}



RcppExport
void
mlp_grad(const int *lays, const int *no_lays, const int *n_pts,
         const int *w_pts, const int *w_fl, const double *w_val,
         const int *af, const double *af_p,
         const int *no_datarows, const double *in, const double *out,
         double *msegrad)
{
    try {
        double mse = fcnn::internal::grad(lays, *no_lays, n_pts,
                                        w_pts, w_fl, w_val,
                                        af, af_p,
                                        *no_datarows, in, out, msegrad + 1);
        msegrad[0] = mse;
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }
}



RcppExport
void
mlp_gradi(const int *lays, const int *no_lays, const int *n_pts,
          const int *w_pts, const int *w_fl, const double *w_val,
          const int *af, const double *af_p,
          const int *no_datarows, const int *i, double *in, const double *out,
          double *grad)
{
    try {
        fcnn::internal::gradi(lays, *no_lays, n_pts,
                            w_pts, w_fl, w_val,
                            af, af_p,
                            *no_datarows, *i - 1, in, out, grad);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }
}



RcppExport
void
mlp_gradij(const int *lays, const int *no_lays, const int *n_pts,
           const int *w_pts, const int *w_fl, const double *w_val, const int *no_w_on,
           const int *af, const double *af_p,
           const int *no_datarows, const int *i, double *in,
           double *grad)
{
    try {
        fcnn::internal::gradij(lays, *no_lays, n_pts,
                            w_pts, w_fl, w_val, *no_w_on,
                            af, af_p,
                            *no_datarows, *i - 1, in, grad);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }
}



RcppExport
void
mlp_jacob(const int *lays, const int *no_lays, const int *n_pts,
          const int *w_pts, const int *w_fl, const double *w_val, const int *no_w_on,
          const int *af, const double *af_p,
          const int *no_datarows, const int *i, double *in,
          double *jacob)
{
    try {
        fcnn::internal::jacob(lays, *no_lays, n_pts,
                              w_pts, w_fl, w_val, *no_w_on,
                              af, af_p,
                              *no_datarows, *i - 1, in, jacob);
    } catch (fcnn::exception &e) {
        Rf_error(e.what());
    }
}



RcppExport
void
ihessupdate(const int *nw, const int *no, double *a, const double *g, double *H)
{
    fcnn::internal::ihessupdate(*nw, *no, *a, g, H);
}
















