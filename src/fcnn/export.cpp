/*
 *  This file is a part of Fast Compressed Neural Networks.
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

/** \file export.cpp
 *  \brief Exporting networks.
 */


#include <fcnn/export.h>
#include <fcnn/activation.h>
#include <fcnn/utils.h>
#include <fstream>
#include <iomanip>
#include <cctype>


using namespace fcnn;
using namespace fcnn::internal;



namespace {

template <typename TA, typename TB>
struct types_eq {
    static const bool val = false;
};

template <typename T>
struct types_eq<T, T> {
    static const bool val = true;
};


template <typename T>
std::string mlp_act_f_C_code(int af, T s, const std::string &t,
                             const std::string &x, const std::string &y,
                             const std::string &tab)
{
    switch (af) {
        case threshold:
            return tab + y + " = (" + x + " >= 0) ? (" + t + ")1 : (" + t + ")0;\n";
        case sym_threshold:
            return tab + y + " = (" + x + " >= 0) ? (" + t + ")1 : (" + t + ")-1;\n";
        case linear:
            return tab + y + " = (" + t + ")" + num2str(s) + " * " + x + ";\n";
        case sigmoid:
            return tab + y + " = (" + t + ")1 / ((" + t + ")1 + exp((" + t + ")"
                   + num2str(-2 * s) + " * " + x + "));\n";
        case sym_sigmoid:
            return tab + y + "(" + t + ")2. / ((" + t + ")1. + exp((" + t + ")"
                   + num2str(-2. * s) + " * x)) - (" + t + ")1.";
        case sigmoid_approx:
            return tab + x + " = (" + t + ")" + num2str(s) + " * " + x + ";\n"
                   + tab + "if (" + x + " < (" + t + ")-2.8) " + y
                   + " = (" + t + ")2. / ((" + t + ")1. + exp((" + t + ")-2. * " + x + ")) - (" + t + ")1.;\n"
                   + tab + "else if (" + x + " > (" + t + ") 2.8) " + y
                   + " = (" + t + ")2. / ((" + t + ")1. + exp((" + t + ")-2. * " + x + ")) - (" + t + ")1.;\n"
                   + tab + "else if (" + x + " < (" + t + ")-2.0) " + y
                   + " = (" + t + ")0.03575493 * " + x + " - (" + t + ")0.8925177;\n"
                   + tab + "else if (" + x + " < (" + t + ")-1.2) " + y
                   + " = (" + t + ")0.16296622 * " + x + " - (" + t + ")0.6380951;\n"
                   + tab + "else if (" + x + " < (" + t + ")-0.7) " + y
                   + " = (" + t + ")0.45857366 * " + x + " - (" + t + ")0.2833662;\n"
                   + tab + "else if (" + x + " < (" + t + ")0.7) " + y
                   + " = (" + t + ")0.86338254 * " + x + ";\n"
                   + tab + "else if (" + x + " < (" + t + ")1.2) " + y
                   + " = (" + t + ")0.45857366 * " + x + " + (" + t + ")0.2833662;\n"
                   + tab + "else if (" + x + " < (" + t + ")2.0) " + y
                   + " = (" + t + ")0.16296622 * " + x + " + (" + t + ")0.6380951;\n"
                   + tab + "else " + y + " = (" + t + ")0.03575493 * " + x + " + (" + t + ")0.8925177;\n"
                   + tab + y + " = (" + t + ")0.5 + (" + t + ")0.5 * " + y + ";\n";
        case sym_sigmoid_approx:
            return tab + x + " = (" + t + ")" + num2str(s) + " * " + x + ";\n"
                   + tab + "if (" + x + " < (" + t + ")-2.8) " + y
                   + " = (" + t + ")2. / ((" + t + ")1. + exp((" + t + ")-2. * " + x + ")) - (" + t + ")1.;\n"
                   + tab + "else if (" + x + " > (" + t + ")2.8) " + y
                   + " = (" + t + ")2. / ((" + t + ")1. + exp((" + t + ")-2. * " + x + ")) - (" + t + ")1.;\n"
                   + tab + "else if (" + x + " < (" + t + ")-2.0) " + y
                   + " = (" + t + ")0.03575493 * " + x + " - (" + t + ")0.8925177;\n"
                   + tab + "else if (" + x + " < (" + t + ")-1.2) " + y
                   + " = (" + t + ")0.16296622 * " + x + " - (" + t + ")0.6380951;\n"
                   + tab + "else if (" + x + " < (" + t + ")-0.7) " + y
                   + " = (" + t + ")0.45857366 * " + x + " - (" + t + ")0.2833662;\n"
                   + tab + "else if (" + x + " < (" + t + ")0.7) " + y
                   + " = (" + t + ")0.86338254 * " + x + ";\n"
                   + tab + "else if (" + x + " < (" + t + ")1.2) " + y
                   + " = (" + t + ")0.45857366 * " + x + " + (" + t + ")0.2833662;\n"
                   + tab + "else if (" + x + " < (" + t + ")2.0) " + y
                   + " = (" + t + ")0.16296622 * " + x + " + (" + t + ")0.6380951;\n"
                   + tab + "else " + y + " = (" + t + ")0.03575493 * " + x + " + (" + t + ")0.8925177;\n";
        default:
            error("invalid activation function id");
    }
    return "";
}

} /* namespace */





template <typename T>
bool
fcnn::internal::mlp_export_C(const std::string &fname,
                             const std::string &netname,
                             const std::vector<int> &layers,
                             const std::vector<int> &n_p,
                             const std::vector<T> &w_val,
                             int hl_af, T hl_af_p, int ol_af, T ol_af_p)
{
    std::ofstream file(fname.c_str());
    if (!file.good()) return false;

    std::string name;
    if (!netname[0]) {
        name = "exported_net";
    } else {
        int i = 0, n = netname.length();
        for (; i < n; ++i) {
            char c = netname[i];
            if (std::isalnum(c)) name += c;
            else name += '_';
        }
    }

    file << "/*\n * This file was generated automatically by FCNN on "
         << time_str() << '\n';
    file << " * Exported network C function name: " << name << "\n";
    file << " */\n\n\n";
    file << "#include <math.h>\n\n\n";

    std::string tname;
    if (types_eq<T, float>::val) tname = "float";
    if (types_eq<T, double>::val) tname = "double";
    file << "void\n" << name << "(const " << tname << " *input, "
         << tname << " *output)\n{\n";

    int nol = layers.size(), non = n_p[nol], now = w_val.size();
    std::string tab;
    if (types_eq<T, float>::val) tab = "                 ";
    if (types_eq<T, double>::val) tab = "                  ";
    file << "    " << tname << " n[" << non << "];\n";
    file << "    " << tname << " w[] = {";
    file << std::setprecision(precision<T>::val);
    for (int i = 0; i < now; ++i) {
        file << ' ' << w_val[i];
        if (i == (now - 1)) file << " };\n";
        else {
            file << ',';
            if (!((i + 1) % 4)) file << '\n' << tab;
        }
    }
    file << "    " << tname << " x;\n";
    file << "    int wi = 0, ni = " << n_p[1] << ", np;\n";

    tab = "    ";
    file << '\n';
    file << tab << "for (; ni < " << n_p[2] << "; ++ni) {\n";
    file << tab << tab << "x = w[wi++];\n";
    file << tab << tab << "for (np = 0; np < "
            << n_p[1] << "; ++np) x += input[np] * w[wi++];\n";
    file << mlp_act_f_C_code(hl_af, hl_af_p, tname, "x", "n[ni]", tab + tab);
    file << tab << "}\n";
    int l = 2;
    for (; l < nol - 1; ++l) {
        file << tab << "for (; ni < " << n_p[l + 1] << "; ++ni) {\n";
        file << tab << tab << "x = w[wi++];\n";
        file << tab << tab << "for (np = "<< n_p[l - 1] << "; np < "
             << n_p[l] << "; ++np) x += n[np] * w[wi++];\n";
        file << mlp_act_f_C_code(hl_af, hl_af_p, tname, "x", "n[ni]", tab + tab);
        file << tab << "}\n";
    }
    file << tab << "for (ni = 0; ni < " << (n_p[l + 1] - n_p[l]) << "; ++ni) {\n";
    file << tab << tab << "x = w[wi++];\n";
    file << tab << tab << "for (np = "<< n_p[l - 1] << "; np < "
            << n_p[l] << "; ++np) x += n[np] * w[wi++];\n";
    file << mlp_act_f_C_code(ol_af, ol_af_p, tname, "x", "output[ni]", tab + tab);
    file << tab << "}\n";

    file << "}\n\n";

    if (!file.good()) return false;
    return true;
}


namespace {



template <typename T>
void write_transf(std::ofstream &file, const std::string &name,
                  const std::string &tname,
                  const T *A, const T *b, int N)
{
    file << "void\n" << name << "(const " << tname << " *x, "
         << tname << " *y)\n{\n";
    for (int i = 0; i < N; ++i) {
        file << "    y[" << i << "] = ";
        int k = 0;
        for (int j = 0; j < N; ++j) {
            T a = A[i + j * N];
            if (a != T()) {
                if (k && !(k % 2)) file << "\n             ";
                if (k) file << " + ";
                ++k;
                if (a != (T)1) {
                    file << "(" << tname << ")" << a << " * x[" << j << "]";
                } else {
                    file << "x[" << j << "]";
                }
            }
        }
        if (b[i] != T()) {
            if (k && !(k % 2)) file << "\n             ";
            if (k) file << " + ";
            ++k;
            file << "(" << tname << ")" << b[i];
        }
        if (k) file << ";\n"; else file << "0;\n";
    }
    file << "}\n\n\n";
}


} /* namespace */





template <typename T>
bool
fcnn::internal::mlp_export_C(const std::string &fname,
                             const std::string &netname,
                             const std::vector<int> &layers,
                             const std::vector<int> &n_p,
                             const std::vector<T> &w_val,
                             int hl_af, T hl_af_p, int ol_af, T ol_af_p,
                             const T *A, const T *b, const T *C, const T *d)
{
    std::ofstream file(fname.c_str());
    if (!file.good()) return false;

    std::string name;
    if (!netname[0]) {
        name = "exported_net";
    } else {
        int i = 0, n = netname.length();
        for (; i < n; ++i) {
            char c = netname[i];
            if (std::isalnum(c)) name += c;
            else name += '_';
        }
    }

    file << "/*\n * This file was generated automatically by FCNN on "
         << time_str() << '\n';
    file << " * Exported network C function name: " << name << "\n";
    file << " */\n\n\n";
    file << "#include <math.h>\n\n\n";

    std::string tname;
    if (types_eq<T, float>::val) tname = "float";
    if (types_eq<T, double>::val) tname = "double";

    int nol = layers.size(), non = n_p[nol], now = w_val.size();

    write_transf(file, name + "_in", tname, A, b, layers[0]);
    write_transf(file, name + "_out", tname, C, d, layers[nol - 1]);

    file << "void\n" << name << "(const " << tname << " *input, "
         << tname << " *output)\n{\n";

    std::string tab;
    if (types_eq<T, float>::val) tab = "                 ";
    if (types_eq<T, double>::val) tab = "                  ";
    file << "    " << tname << " n[" << non << "];\n";
    file << "    " << tname << " w[] = {";
    file << std::setprecision(precision<T>::val);
    for (int i = 0; i < now; ++i) {
        file << ' ' << w_val[i];
        if (i == (now - 1)) file << " };\n";
        else {
            file << ',';
            if (!((i + 1) % 4)) file << '\n' << tab;
        }
    }
    file << "    " << tname << " x;\n";
    file << "    int wi = 0, ni = " << n_p[1] << ", np;\n";

    file << '\n';
    tab = "    ";
    file << tab << name << "_in(input, n);\n";
    int l = 1;
    for (; l < nol - 1; ++l) {
        file << tab << "for (; ni < " << n_p[l + 1] << "; ++ni) {\n";
        file << tab << tab << "x = w[wi++];\n";
        file << tab << tab << "for (np = "<< n_p[l - 1] << "; np < "
             << n_p[l] << "; ++np) x += n[np] * w[wi++];\n";
        file << mlp_act_f_C_code(hl_af, hl_af_p, tname, "x", "n[ni]", tab + tab);
        file << tab << "}\n";
    }
    file << tab << "for (; ni < " << n_p[l + 1] << "; ++ni) {\n";
    file << tab << tab << "x = w[wi++];\n";
    file << tab << tab << "for (np = "<< n_p[l - 1] << "; np < "
            << n_p[l] << "; ++np) x += n[np] * w[wi++];\n";
    file << mlp_act_f_C_code(ol_af, ol_af_p, tname, "x", "n[ni]", tab + tab);
    file << tab << "}\n";
    file << tab << name << "_out(n + " << n_p[l] << ", output);\n";

    file << "}\n\n";

    if (!file.good()) return false;
    return true;
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template bool fcnn::internal::mlp_export_C(const std::string &fname,
                                           const std::string &netname,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           int, float, int, float);
template bool fcnn::internal::mlp_export_C(const std::string &fname,
                                           const std::string &netname,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           int, float, int, float,
                                           const float*, const float*,
                                           const float*, const float*);
#endif /* FCNN_DOUBLE_ONLY */
template bool fcnn::internal::mlp_export_C(const std::string &fname,
                                           const std::string &netname,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           int, double, int, double);
template bool fcnn::internal::mlp_export_C(const std::string &fname,
                                           const std::string &netname,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           int, double, int, double,
                                           const double*, const double*,
                                           const double*, const double*);


