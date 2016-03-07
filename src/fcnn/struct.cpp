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

/** \file struct.cpp
 *  \brief Working with multilayer perceptron network structure.
 */


#include <fcnn/struct.h>
#include <fcnn/activation.h>
#include <fcnn/error.h>
#include <fcnn/utils.h>
#include <fcnn/report.h>
#include <set>
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
void
fcnn::internal::mlp_expand_reorder_inputs(std::vector<int> &layers,
                                          std::vector<int> &n_p,
                                          std::vector<int> &n_prev,
                                          std::vector<int> &n_next,
                                          std::vector<int> &w_p,
                                          std::vector<T> &w_val,
                                          std::vector<int> &w_fl,
                                          int newnoinp,
                                          const std::map<int, int> &m)
{
    // Check new no of inputs
    int ninp = newnoinp - layers[0];
    if (ninp < 0)
        throw exception("number of inputs in the new network must not be \
smaller than the number of inputs in the initial network");
    // Fill missing values in mapping
    std::map<int, int> mcpy(m);
    if (mcpy.size() < layers[0]) {
        for (int i = 1; i <= layers[0]; ++i) {
            if (mcpy.find(i) == mcpy.end())
                mcpy[i] = i;
        }
    }
    // Check mapping / construct inverse mapping
    std::map<int, int>::const_iterator it;
    std::set<int> mv;
    for (it = mcpy.begin(); it != mcpy.end(); ++it) {
        int frst = it->first;
        int scnd = it->second;
        if ((scnd < 1) || (scnd > newnoinp) || (frst < 1) || (frst > layers[0]))
            throw exception("invalid mapping of inputs (index out of range)");
        if (!mv.insert(scnd).second)
            throw exception("invalid mapping of inputs (duplicated or ambiguous indices)");
    }
    // Expand inputs (place old ones at the beginning)
    if (ninp) {
        int nol = layers.size(), l0p1 = layers[0] + 1;
        // Layers, neurons
        layers[0] += ninp;
        n_prev.insert(n_prev.begin() + n_p[1], ninp, 0);
        n_next.insert(n_next.begin() + n_p[1], ninp, 0);
        for (int ll = 1; ll <= nol; ++ll) n_p[ll] += ninp;
        // Weights
        for (int n = 0, wp = w_p[2]; n < layers[1]; ++n, wp -= l0p1) {
            w_val.insert(w_val.begin() + wp, ninp, T());
            w_fl.insert(w_fl.begin() + wp, ninp, 0);
        }
        for (int ll = 2; ll <= nol; ++ll) w_p[ll] += layers[1] * ninp;
    }
    // Temporary vectors
    std::vector<int> nnext(n_p[1]);
    std::vector<T> nval(w_p[2]);
    std::vector<int> nfl(w_p[2]);
    int l0p1 = layers[0] + 1;
    for (int n = 0, w = 0; n < layers[1]; ++n, w += l0p1) {
        nval[w] = w_val[w];
        nfl[w] = w_fl[w];
    }
    // Reorder connections
    for (it = mcpy.begin(); it != mcpy.end(); ++it) {
        int id1 = it->first, id2 = it->second;
        nnext[id2 - 1] = n_next[id1 - 1];
        for (int n = 0, w1 = id1, w2 = id2; n < layers[1];
                ++n, w1 += l0p1, w2 += l0p1) {
            nval[w2] = w_val[w1];
            nfl[w2] = w_fl[w1];
        }
    }
    // Finalise
    for (int i = 0; i < n_p[1]; ++i) n_next[i] = nnext[i];
    for (int i = 0; i < w_p[2]; ++i) {
        w_val[i] = nval[i];
        w_fl[i] = nfl[i];
    }
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::mlp_expand_reorder_inputs(std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<float>&,
                                                        std::vector<int>&,
                                                        int,
                                                        const std::map<int, int>&);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::mlp_expand_reorder_inputs(std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<int>&,
                                                        std::vector<double>&,
                                                        std::vector<int>&,
                                                        int,
                                                        const std::map<int, int>&);



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
                               std::vector<int> & af,
                               std::vector<T> &af_p,
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
                T in = mlp_act_f(af[l], af_p[l], w_val[bi]);
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
            --layers[l];
            if (report) {
                message mes;
                mes << "removing neuron " << (n - n_p[l] + 1)
                    << " in layer " << (l + 1) << " ("  << layers[l];
                if (layers[l] == 1) mes << " neuron remains in this layer; ";
                else mes << " neurons remain in this layer, ";
                mes << (n_p[nol] - 1) << " total)";
                fcnn::internal::report(mes);
            }
            for (int ll = l + 1; ll <= nol; ++ll) --n_p[ll];
            --ni; --n;
            ++count;
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
                                            std::vector<int>&, std::vector<float>&,
                                            bool);
#endif /* FCNN_DOUBLE_ONLY */
template int fcnn::internal::mlp_rm_neurons(std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<int>&,
                                            std::vector<int>&, std::vector<double>&,
                                            std::vector<int>&, int&,
                                            std::vector<int>&, std::vector<double>&,
                                            bool);


template <typename T>
void
fcnn::internal::mlp_rm_input_neurons(std::vector<int> &layers,
                                     std::vector<int> &n_p,
                                     std::vector<int> &n_prev,
                                     std::vector<int> &n_next,
                                     std::vector<int> &w_p,
                                     std::vector<T> &w_val,
                                     std::vector<int> &w_fl,
                                     bool report)
{
    int nol = layers.size();
    std::vector<int> rm;
    for (int n = 0; n < n_p[1]; ++n) {
        if (n_next[n]) continue;
        // remove connections to this neuron
        int wi = w_p[1] + n + 1;
        for (int nn = 0; nn < layers[1]; ++nn, wi += layers[0]) {
            w_val.erase(w_val.begin() + wi);
            w_fl.erase(w_fl.begin() + wi);
        }
        for (int ll = 2; ll <= nol; ++ll) w_p[ll] -= layers[1];
        // remove neuron
        if (report) rm.push_back(rm.size() + n + 1);
        n_prev.erase(n_prev.begin() + n);
        n_next.erase(n_next.begin() + n);
        --layers[0];
        for (int ll = 1; ll <= nol; ++ll) --n_p[ll];
        --n;
    }
    if (report && rm.size()) {
        message mes;
        if (rm.size() > 1) {
            mes << "removed neurons";
            for (std::vector<int>::const_iterator it = rm.begin(); it != rm.end(); ++it)
                mes << " " << *it;
        } else {
            mes << "removed neuron " << rm[0];
        }
        mes << " in the input layer ("  << layers[0];
        if (layers[0] == 1) mes << " neuron remains in this layer; ";
        else mes << " neurons remain in this layer)";
        fcnn::internal::report(mes);
    }
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template void fcnn::internal::mlp_rm_input_neurons(std::vector<int>&, std::vector<int>&,
                                                   std::vector<int>&, std::vector<int>&,
                                                   std::vector<int>&, std::vector<float>&,
                                                   std::vector<int>&, bool);
#endif /* FCNN_DOUBLE_ONLY */
template void fcnn::internal::mlp_rm_input_neurons(std::vector<int>&, std::vector<int>&,
                                                   std::vector<int>&, std::vector<int>&,
                                                   std::vector<int>&, std::vector<double>&,
                                                   std::vector<int>&, bool);





template <typename T>
void
fcnn::internal::mlp_merge(const std::vector<int> &Alayers,
                          const std::vector<int> &Aw_p,
                          const std::vector<T> &Aw_val,
                          const std::vector<int> &Aw_fl,
                          const std::vector<int> &Blayers,
                          const std::vector<int> &Bw_p,
                          const std::vector<T> &Bw_val,
                          const std::vector<int> &Bw_fl,
                          bool same_inputs,
                          std::vector<int> &layers,
                          std::vector<int> &n_p,
                          std::vector<int> &n_prev,
                          std::vector<int> &n_next,
                          std::vector<int> &w_p,
                          std::vector<T> &w_val,
                          std::vector<int> &w_fl,
                          int &w_on)
{
    // layers
    int nol = Alayers.size();
    if (nol != Blayers.size())
        throw exception("numbers of layers disagree");
    layers.resize(nol);
    if (same_inputs) {
        if (Alayers[0] != Blayers[0])
            throw exception("numbers of neurons in the input layers disagree");
        layers[0] = Alayers[0];
    } else {
        layers[0] = Alayers[0] + Blayers[0];
    }
    for (int i = 1; i < nol; ++i)
        layers[i] = Alayers[i] + Blayers[i];

    // neurons
    n_p.assign(nol + 1, 0);
    for (int i = 0; i < nol; ++i) {
        n_p[i + 1] = n_p[i] + layers[i];
    }

    // weights
    w_p.assign(nol + 1, 0);
    for (int i = 1; i < nol; ++i)
        w_p[i + 1] = w_p[i] + layers[i] * (layers[i - 1] + 1);
    w_val.assign(w_p[nol], 0.);
    w_fl.assign(w_p[nol], 0);

    // weights, active connection count, and neuron connection info
    w_on = 0;
    n_prev.assign(n_p[nol], 0);
    n_next.assign(n_p[nol], 0);
    int l = 0;
    if (same_inputs) {
        l = 1;
        int wi = w_p[l], ni = n_p[l];
        for (int niA = 0, wiA = Aw_p[l]; niA < Alayers[l]; ++niA, ++ni) {
            w_fl[wi] = Aw_fl[wiA];
            if (w_fl[wi]) {
                w_val[wi] = Aw_val[wiA];
                ++w_on;
            }
            ++wi; ++wiA;
            for (int npi = 0; npi < layers[0]; ++npi, ++wiA, ++wi) {
                w_fl[wi] = Aw_fl[wiA];
                if (w_fl[wi]) {
                    ++w_on;
                    ++n_prev[ni];
                    ++n_next[npi];
                    w_val[wi] = Aw_val[wiA];
                }
            }
        }
        for (int niB = 0, wiB = Bw_p[l]; niB < Blayers[l]; ++niB, ++ni) {
            w_fl[wi] = Bw_fl[wiB];
            if (w_fl[wi]) {
                w_val[wi] = Bw_val[wiB];
                ++w_on;
            }
            ++wiB; ++wi;
            for (int npi = 0; npi < layers[0]; ++npi, ++wiB, ++wi) {
                w_fl[wi] = Bw_fl[wiB];
                if (w_fl[wi]) {
                    ++w_on;
                    ++n_prev[ni];
                    ++n_next[npi];
                    w_val[wi] = Bw_val[wiB];
                }
            }
        }
    }
    for (++l; l < nol; ++l) {
        int wi = w_p[l], ni = n_p[l];
        for (int niA = 0, wiA = Aw_p[l]; niA < Alayers[l];
             ++niA, ++ni, wi += Blayers[l - 1]) {
            w_fl[wi] = Aw_fl[wiA];
            if (w_fl[wi]) {
                w_val[wi] = Aw_val[wiA];
                ++w_on;
            }
            ++wi; ++wiA;
            for (int npiA = 0, npi = n_p[l - 1]; npiA < Alayers[l - 1];
                 ++npiA, ++npi, ++wiA, ++wi) {
                w_fl[wi] = Aw_fl[wiA];
                if (w_fl[wi]) {
                    ++w_on;
                    ++n_prev[ni];
                    ++n_next[npi];
                    w_val[wi] = Aw_val[wiA];
                }
            }
        }
        wi = w_p[l] + Alayers[l] * (layers[l - 1] + 1), ni = n_p[l] + Alayers[l];
        for (int niB = 0, wiB = Bw_p[l]; niB < Blayers[l];
             ++niB, ++ni) {
            w_fl[wi] = Bw_fl[wiB];
            if (w_fl[wi]) {
                w_val[wi] = Bw_val[wiB];
                ++w_on;
            }
            ++wiB;
            wi += Alayers[l - 1] + 1;
            for (int npiB = 0, npi = n_p[l - 1] + Alayers[l - 1];
                 npiB < Blayers[l - 1]; ++npiB, ++npi, ++wiB, ++wi) {
                w_fl[wi] = Bw_fl[wiB];
                if (w_fl[wi]) {
                    ++w_on;
                    ++n_prev[ni];
                    ++n_next[npi];
                    w_val[wi] = Bw_val[wiB];
                }
            }
        }
    }
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template
void
fcnn::internal::mlp_merge(const std::vector<int>&, const std::vector<int>&,
                          const std::vector<float>&, const std::vector<int>&,
                          const std::vector<int>&, const std::vector<int>&,
                          const std::vector<float>&, const std::vector<int>&,
                          bool,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<float>&,
                          std::vector<int>&, int&);
#endif /* FCNN_DOUBLE_ONLY */
template
void
fcnn::internal::mlp_merge(const std::vector<int>&, const std::vector<int>&,
                          const std::vector<double>&, const std::vector<int>&,
                          const std::vector<int>&, const std::vector<int>&,
                          const std::vector<double>&, const std::vector<int>&,
                          bool,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<double>&,
                          std::vector<int>&, int&);




template <typename T>
void
fcnn::internal::mlp_stack(const std::vector<int> &Alayers, const std::vector<int> &Aw_p,
                          const std::vector<T> &Aw_val, const std::vector<int> &Aw_fl,
                          const std::vector<int> &Blayers, const std::vector<int> &Bw_p,
                          const std::vector<T> &Bw_val, const std::vector<int> &Bw_fl,
                          std::vector<int> &layers, std::vector<int> &n_p,
                          std::vector<int> &n_prev, std::vector<int> &n_next,
                          std::vector<int> &w_p, std::vector<T> &w_val,
                          std::vector<int> &w_fl, int &w_on)
{
    // layers
    if (Alayers.back() != Blayers.front())
        throw exception("numbers of output and input neurons in stacked networks disagree");
    layers = Alayers;
    layers.insert(layers.end(), Blayers.begin() + 1, Blayers.end());
    int nol = layers.size();

    // neurons
    n_p.assign(nol + 1, 0);
    for (int i = 0; i < nol; ++i) n_p[i + 1] = n_p[i] + layers[i];

    // weights
    w_p = Aw_p;
    w_p.insert(w_p.end(), Bw_p.begin() + 2, Bw_p.end());
    for (int i = Alayers.size() + 1; i <= nol; ++i) w_p[i] += Aw_p.back();
    w_val = Aw_val;
    w_val.insert(w_val.end(), Bw_val.begin(), Bw_val.end());
    w_fl = Aw_fl;
    w_fl.insert(w_fl.end(), Bw_fl.begin(), Bw_fl.end());

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
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template
void
fcnn::internal::mlp_stack(const std::vector<int>&, const std::vector<int>&,
                          const std::vector<float>&, const std::vector<int>&,
                          const std::vector<int>&, const std::vector<int>&,
                          const std::vector<float>&, const std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<float>&,
                          std::vector<int>&, int&);
#endif /* FCNN_DOUBLE_ONLY */
template
void
fcnn::internal::mlp_stack(const std::vector<int>&, const std::vector<int>&,
                          const std::vector<double>&, const std::vector<int>&,
                          const std::vector<int>&, const std::vector<int>&,
                          const std::vector<double>&, const std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<int>&,
                          std::vector<int>&, std::vector<double>&,
                          std::vector<int>&, int&);



namespace {

template <typename TA, typename TB>
struct types_eq {
    static const bool val = false;
};

template <typename T>
struct types_eq<T, T> {
    static const bool val = true;
};

} /* namespace */


template <typename T>
bool
fcnn::internal::mlp_save_txt(const std::string &fname,
                             const std::string &netname,
                             const std::vector<int> &layers,
                             const std::vector<T> &w_val,
                             const std::vector<int> &w_fl,
                             const std::vector<int> &af,
                             const std::vector<T> &af_p)
{
    std::ofstream file(fname.c_str());
    if (!file.good()) return false;
    if (netname[0]) write_comment(file, netname);
    else file << "#\n";

    file << "\n# FCNN " << fcnn_ver() << " network representation saved on "
         << time_str() << "\n\n";

    file << "# layers (" << num2str((unsigned)layers.size()) << ")\n";
    for (int i = 0; i < (int)layers.size(); ++i) {
        if (i) file << ' ';
        file << layers[i];
    }
    file << "\n\n# flags (" << num2str((unsigned)w_fl.size()) << ")\n";
    int awcount = 0;
    for (int i = 0, n = w_fl.size(); i < n; ++i) {
        if (i % 40) file << ' '; else if (i) file << '\n';
        if (w_fl[i]) { file << '1'; ++awcount; } else file << '0';
    }
    file << "\n\n# weights (" << num2str(awcount) << ")\n";
    file << std::setprecision(precision<T>::val);
    int noinrow;
    if (types_eq<T, float>::val) noinrow = 8;
    if (types_eq<T, double>::val) noinrow = 4;
    for (int i = 0, k = 0, n = w_fl.size(); i < n; ++i) {
        if (w_fl[i]) {
            if (k % noinrow) file << ' '; else if (k) file << '\n';
            file << w_val[i];
            ++k;
        }
    }
    file << "\n\n# activation functions\n";
    for (int i = 1; i < (int)layers.size(); ++i) {
        file << "# layer " << (i + 1);
        if (i < layers.size() - 1) file << " (hidden " << i << ") : ";
        else file << " (output) : ";
        file << mlp_act_f_str(af[i]);
        if (af[i] > 2) file << " with s = " << num2str(af_p[i]);
        file << '\n';
    }
    for (int i = 1; i < (int)layers.size(); ++i) {
        file << af[i] << ' ' << af_p[i] << '\n';
    }
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
                             std::vector<int> &af,
                             std::vector<T> &af_p)
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
    while (file && !is_deol(file)) {
        int fl;
        if (read(file, fl)) {
            if ((fl == 0) || (fl == 1)) w_fl.push_back(fl);
            else return false;
        }
        else return false;
    }
    if (!file) return false;

    skip_all(file);
    std::vector<T> vals;
    while (!file.fail() && !is_deol(file)) {
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
    if (file.eof()) return false;
    af.push_back(0);
    af_p.push_back((T)0);
    while (!file.fail() && !is_deol(file)) {
        int a;
        T ap;
        if (read(file, a)) af.push_back(a);
        if (is_eol(file)) return false;
        if (read(file, ap)) af_p.push_back(ap);
        if (ap <= T()) return false;
    }
    if (af.size() != layers.size()) return false;
    skip_all(file);
    if (!file.eof()) return false;

    return true;
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template bool fcnn::internal::mlp_save_txt(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<float>&);
template bool fcnn::internal::mlp_load_txt(const std::string&,
                                           std::string&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<float>&,
                                           std::vector<int>&, int&,
                                           std::vector<int>&,
                                           std::vector<float>&);
#endif /* FCNN_DOUBLE_ONLY */
template bool fcnn::internal::mlp_save_txt(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<double>&);
template bool fcnn::internal::mlp_load_txt(const std::string&,
                                           std::string&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<int>&,
                                           std::vector<int>&, std::vector<double>&,
                                           std::vector<int>&, int&,
                                           std::vector<int>&,
                                           std::vector<double>&);






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
















