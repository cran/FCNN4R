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

/** \file struct.cpp
 *  \brief Working with multilayer perceptron network structure.
 */


#include <fcnn/struct.h>
#include <fcnn/activation.h>
#include <fcnn/error.h>
#include <fcnn/utils.h>
#include <fcnn/report.h>
#include <fstream>
#include <iomanip>
#include <cctype>


using namespace fcnn;
using namespace fcnn::internal;


template <typename T>
void
fcnn::internal::mlp_construct(const std::vector<int> &layers,
                              std::vector<int> &n_p,
                              std::vector<int> &n_prev,
                              std::vector<int> &n_next,
                              std::vector<int> &w_p,
                              std::vector<T> &w_val,
                              std::vector<int> &w_fl,
                              int &w_on)
{
    // layers
    if (layers.size() < 3)
        throw exception("there must be at least 3 layers (input, at least \
one hidden, and output)");
    int nol = layers.size();

    // neurons
    n_p.assign(nol + 1, 0);
    for (int i = 0; i < nol; ++i) {
        if (layers[i] < 1)
            throw exception("invalid no of neurons (" + num2str(layers[i])
                                  + ") in layer " + num2str(i + 1));
        n_p[i + 1] = n_p[i] + layers[i];
    }
    n_prev.assign(n_p[nol], 0);
    n_next.assign(n_p[nol], 0);
    for (int j = n_p[0], n = n_p[1]; j < n; ++j) {
        n_next[j] = layers[1];
    }
    for (int i = 1; i < nol - 1; ++i) {
        for (int j = n_p[i], n = n_p[i + 1]; j < n; ++j) {
            n_next[j] = layers[i + 1];
            n_prev[j] = layers[i - 1];
        }
    }
    for (int j = n_p[nol - 1], n = n_p[nol]; j < n; ++j) {
        n_prev[j] = layers[nol - 2];
    }

    // weights
    w_p.assign(nol + 1, 0);
    for (int i = 1; i < nol; ++i)
        w_p[i + 1] = w_p[i] + layers[i] * (layers[i - 1] + 1);
    w_on = w_p[nol];
    w_val.assign(w_p[nol], (T)0.);
    w_fl.assign(w_p[nol], (int) true);
}



template <typename T>
void
fcnn::internal::mlp_construct(const std::vector<int> &layers,
                              std::vector<int> &n_p,
                              std::vector<int> &n_prev,
                              std::vector<int> &n_next,
                              std::vector<int> &w_p,
                              const std::vector<T> &w_act_val,
                              std::vector<T> &w_val,
                              const std::vector<int> &w_fl,
                              int &w_on)
{
    // layers
    if (layers.size() < 3)
        throw exception("there must be at least 3 layers (input, at least \
one hidden, and output)");
    int nol = layers.size();

    // neurons
    n_p.assign(nol + 1, 0);
    for (int i = 0; i < nol; ++i) {
        if (layers[i] < 1)
            throw exception("invalid no of neurons ("
                                  + num2str(layers[i]) + ") in layer "
                                  + num2str(i + 1));
        n_p[i + 1] = n_p[i] + layers[i];
    }

    // weights
    w_p.assign(nol + 1, 0);
    for (int i = 1; i < nol; ++i)
        w_p[i + 1] = w_p[i] + layers[i] * (layers[i - 1] + 1);
    if (w_p[nol] != (int)w_fl.size()) {
        message mes;
        mes << "no. of weights in given topology (" << w_p[nol]
            << ") differs from the no. of flags provided ("
            << (int) w_act_val.size() << ")";
        throw exception(mes);
    }
    w_val.assign(w_p[nol], 0.);

    // update active connection count and neuron connection info
    w_on = 0;
    n_prev.assign(n_p[nol], 0);
    n_next.assign(n_p[nol], 0);
    for (int l = 1, wi = 0; l < nol; ++l) {
        for (int n = n_p[l], nn = n_p[l + 1]; n < nn; ++n) {
            if (w_fl[wi++]) ++w_on;
            for (int np = n_p[l - 1], nnp = n_p[l]; np < nnp; ++np) {
                if (w_fl[wi++]) {
                    ++w_on;
                    ++n_prev[n];
                    ++n_next[np];
                }
            }
        }
    }

    // Weight values
    if (w_on != (int)w_act_val.size()) {
        message mes;
        mes << "no. of active weights (" << w_on
            << ") and weights provided ("
            << (int) w_act_val.size() << ") disagree";
        throw exception(mes);
    }
    for (int i = 0, j = 0, n = w_p[nol]; i < n; ++i)
        if (w_fl[i]) w_val[i] = w_act_val[j++];
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::mlp_construct(const std::vector<int>&,
                                            std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<float>&,
                                            std::vector<int>&, int&);
template void fcnn::internal::mlp_construct(const std::vector<int>&,
                                            std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&,
                                            const std::vector<float>&, std::vector<float>&,
                                            const std::vector<int>&, int&);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::mlp_construct(const std::vector<int>&,
                                            std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<double>&,
                                            std::vector<int>&, int&);
template void fcnn::internal::mlp_construct(const std::vector<int>&,
                                            std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&,
                                            const std::vector<double>&, std::vector<double>&,
                                            const std::vector<int>&, int&);



template <typename T>
int
fcnn::internal::mlp_rm_neurons(std::vector<int> &layers,
                               std::vector<int> &n_p,
                               std::vector<int> &n_prev,
                               std::vector<int> &n_next,
                               std::vector<int> &w_p,
                               std::vector<T> &w_val,
                               std::vector<int> &w_fl,
                               int &w_on,
                               int hl_af, T hl_af_p,
                               bool report)
{
    int count = 0, nol = layers.size();
start:
    bool again = false;
    for (int l = 1; l < nol - 1; ++l) {
        for (int n = n_p[l], ni = 0; n < n_p[l + 1]; ++n, ++ni) {
            if (((n_next[n]) && (n_prev[n])) || (layers[l] == 1)) continue;
            if (n_next[n]) {
                // this neuron's input to next layer neurons (bias through
                // activation function)
                int bi = w_p[l] + ni * (layers[l - 1] + 1);
                T in = mlp_act_f(hl_af, hl_af_p, w_val[bi]);
                // find neurons connected to this one and update their biases
                int wi = w_p[l + 1] + ni + 1, np1 = layers[l] + 1;
                for (int nn = n_p[l + 1], NN = n_p[l + 2], nni = 0;
                     nn < NN; ++nn, wi += np1, ++nni) {
                    if (w_fl[wi]) {
                        --w_on;
                        --n_prev[nn];
                        bi = w_p[l + 1] + nni * (layers[l] + 1);
                        if (w_fl[bi]) w_val[bi] += in * w_val[wi];
                        else {
                            w_fl[bi] = true;
                            w_val[bi] = in * w_val[wi];
                            ++w_on;
                        }
                    }
                }
            }
            // remove connections to this neuron (all have been handled by now)
            int wi = w_p[l + 1] + ni + 1;
            for (int nn = 0; nn < layers[l + 1]; ++nn, wi += layers[l]) {
                w_val.erase(w_val.begin() + wi);
                w_fl.erase(w_fl.begin() + wi);
            }
            for (int ll = l + 2; ll <= nol; ++ll) w_p[ll] -= layers[l + 1];
            // remove this neuron's connections
            if (n_prev[n]) {
                wi = w_p[l] + ni * (layers[l - 1] + 1) + 1;
                for (int nn = n_p[l - 1], NN = n_p[l];
                     nn < NN; ++nn, ++wi) {
                    if (w_fl[wi]) {
                        --n_next[nn];
                        if (!n_next[nn]) again = true;
                    }
                }
                w_on -= n_prev[n];
            }
            // if bias was on, decrement active weights count
            if (w_fl[w_p[l] + ni * (layers[l - 1] + 1)]) --w_on;
            int wis, wie;
            wis = w_p[l] + ni * (layers[l - 1] + 1);
            wie = w_p[l] + (ni + 1) * (layers[l - 1] + 1);
            w_val.erase(w_val.begin() + wis, w_val.begin() + wie);
            w_fl.erase(w_fl.begin() + wis, w_fl.begin() + wie);
            for (int ll = l + 1; ll <= nol; ++ll)
                w_p[ll] -= (layers[l - 1] + 1);
            // remove neuron
            n_prev.erase(n_prev.begin() + n);
            n_next.erase(n_next.begin() + n);
            for (int ll = l + 1; ll <= nol; ++ll) --n_p[ll];
            --layers[l];
            --ni; --n;
            ++count;
            if (report) {
                message mes;
                mes << "removing neuron " << (int)(n - n_p[l] + 1)
                    << " in layer " << (int)(l + 1) << " ("  << (int)layers[l]
                    << " neurons remain in this layer; "
                    << (int)(n_p[nol]) << " total)";
                fcnn::internal::report(mes);
            }
        }
    }
    if (again) goto start;
    return count;
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template int fcnn::internal::mlp_rm_neurons(std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<float>&,
                                            std::vector<int>&, int&,
                                            int, float, bool);
#endif /* FCNN_DOUBLE_ONLY */
template int fcnn::internal::mlp_rm_neurons(std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<double>&,
                                            std::vector<int>&, int&,
                                            int, double, bool);


template <typename T>
bool
fcnn::internal::mlp_save_txt(const std::string &fname,
                             const std::string &netname,
                             const std::vector<int> &layers,
                             const std::vector<T> &w_val,
                             const std::vector<int> &w_fl,
                             int hl_af, T hl_af_p,
                             int ol_af, T ol_af_p)
{
    std::ofstream file(fname.c_str());
    if (!file.good()) return false;
    if (netname[0]) write_comment(file, netname);
    else file << "#\n";

    file << "\n# saved on " << time_str() << "\n\n";

    file << "# layers\n";
    for (int i = 0; i < (int)layers.size(); ++i) {
        if (i) file << ' ';
        file << layers[i];
    }
    file << "\n# flags\n";
    for (int i = 0, n = w_fl.size(); i < n; ++i) {
        if (i) file << ' ';
        if (w_fl[i]) file << '1'; else file << '0';
    }
    file << "\n# weights\n";
    file << std::setprecision(precision<T>::val);
    for (int i = 0, k = 0, n = w_fl.size(); i < n; ++i) {
        if (w_fl[i]) {
            if (k) file << ' ';
            file << w_val[i];
            ++k;
        }
    }
    file << "\n# activation functions\n";
    file << (int) hl_af << ' ' << hl_af_p << '\n';
    file << (int) ol_af << ' ' << ol_af_p << "\n\n";
    if (file.good()) {
        file.close();
        return true;
    }
    return false;
}




/// Load network in a text file
template <typename T>
bool
fcnn::internal::mlp_load_txt(const std::string &fname,
                             std::string &netname,
                             std::vector<int> &layers,
                             std::vector<int> &n_p,
                             std::vector<int> &n_prev,
                             std::vector<int> &n_next,
                             std::vector<int> &w_p,
                             std::vector<T> &w_val,
                             std::vector<int> &w_fl,
                             int &w_on,
                             int &hl_af, T &hl_af_p,
                             int &ol_af, T &ol_af_p)
{
    std::ifstream file(fname.c_str());
    if (!file.good()) return false;

    skip_blank(file);
    std::string name;
    if (read_comment(file, name)) {
        netname = name;
    }

    skip_all(file);
    while (file && !is_eol(file)) {
        int l;
        if (read(file, l)) layers.push_back(l);
        else return false;
    }
    if (!file) return false;

    skip_all(file);
    while (file && !is_eol(file)) {
        int fl;
        if (read(file, fl)) {
            if (fl == 1) w_fl.push_back(1);
            else if (fl == 0) w_fl.push_back(0);
            else return false;
        }
        else return false;
    }
    if (!file) return false;

    skip_all(file);
    std::vector<T> vals;
    while (!file.fail() && !is_eoleof(file)) {
        T val;
        if (read(file, val)) vals.push_back(val);
        else return false;
    }
    if (file.fail()) return false;

    try {
        mlp_construct(layers, n_p, n_prev, n_next,
                      w_p, vals, w_val, w_fl, w_on);
    } catch (exception &e) {
        return false;
    }

    skip_all(file);
    if (!file.eof()) {
        if (!read(file, hl_af)) return false;
        if (is_eol(file)) return false;
        if (!mlp_act_f_valid(hl_af)) return false;
        if (!read(file, hl_af_p)) return false;
        if (!is_eol(file)) return false;
        if (hl_af_p <= T()) return false;
        if (!read(file, ol_af)) return false;
        if (is_eol(file)) return false;
        if (!mlp_act_f_valid(ol_af)) return false;
        if (!read(file, ol_af_p)) return false;
        if (!is_eoleof(file)) return false;
        if (ol_af_p <= T()) return false;
        skip_all(file);
        if (!file.eof()) return false;
    }

    return true;
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template bool fcnn::internal::mlp_save_txt(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           const std::vector<int>&,
                                           int, float, int, float);
template bool fcnn::internal::mlp_load_txt(const std::string&,
                                           std::string&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<float>&,
                                           std::vector<int>&, int&,
                                           int&, float&, int&, float&);
#endif /* FCNN_DOUBLE_ONLY */
template bool fcnn::internal::mlp_save_txt(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           const std::vector<int>&,
                                           int, double, int, double);
template bool fcnn::internal::mlp_load_txt(const std::string&,
                                           std::string&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<double>&,
                                           std::vector<int>&, int&,
                                           int&, double&, int&, double&);






void
fcnn::internal::mlp_get_lnn_idx(const int *layers, const int *w_p,
                                int i, int &l, int &n, int &npl)
{
    int ll = 0;
    while (w_p[ll] < i) ++ll;
    l = ll;
    --ll;
    int r = i - w_p[ll] - 1;
    n = r / (layers[ll - 1] + 1) + 1;
    npl = r % (layers[ll - 1] + 1);
}


int
fcnn::internal::mlp_get_abs_w_idx(const int *w_fl, int i)
{
    int j = 0, k = 0;
    while (j < i) { if (w_fl[k++]) j++; }
    return k;
}



template <typename T>
void
fcnn::internal::mlp_set_active(const int *layers, const int *n_p,
                               int *n_prev, int *n_next,
                               const int *w_p, T *w_val,
                               int *w_fl, int *w_on,
                               int l, int n, int npl, bool on)
{
    int i = mlp_get_w_idx(layers, w_p, l, n, npl);

    if ((w_fl[i]) && (!on)) {
        w_fl[i] = on;
        w_val[i] = T();
        --*w_on;
        if (npl) {
            --n_next[mlp_get_n_idx(n_p, l - 1, npl)];
            --n_prev[mlp_get_n_idx(n_p, l, n)];
        }
    } else if ((!w_fl[i]) && (on)) {
        w_fl[i] = on;
        ++*w_on;
        if (npl) {
            ++n_next[mlp_get_n_idx(n_p, l - 1, npl)];
            ++n_prev[mlp_get_n_idx(n_p, l, n)];
        }
    }
}



template <typename T>
void
fcnn::internal::mlp_set_active(const int *layers, const int *n_p,
                               int *n_prev, int *n_next,
                               const int *w_p, T *w_val,
                               int *w_fl, int *w_on,
                               int i, bool on)
{
    int ind = i - 1, l, n, npl;
    mlp_get_lnn_idx(layers, w_p, i, l, n, npl);

    if ((w_fl[ind]) && (!on)) {
        w_fl[ind] = on;
        w_val[ind] = T();
        --*w_on;
        if (npl) {
            --n_next[mlp_get_n_idx(n_p, l - 1, npl)];
            --n_prev[mlp_get_n_idx(n_p, l, n)];
        }
    } else if ((!w_fl[ind]) && (on)) {
        w_fl[ind] = on;
        ++*w_on;
        if (npl) {
            ++n_next[mlp_get_n_idx(n_p, l - 1, npl)];
            ++n_prev[mlp_get_n_idx(n_p, l, n)];
        }
    }
}


// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::mlp_set_active(const int*, const int*, int*, int*,
                                             const int*, float*, int*, int*,
                                             int, int, int, bool);
template void fcnn::internal::mlp_set_active(const int*, const int*, int*, int*,
                                             const int*, float*, int*, int*,
                                             int, bool);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::mlp_set_active(const int*, const int*, int*, int*,
                                             const int*, double*, int*, int*,
                                             int, int, int, bool);
template void fcnn::internal::mlp_set_active(const int*, const int*, int*, int*,
                                             const int*, double*, int*, int*,
                                             int, bool);
















