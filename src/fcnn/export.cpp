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
    std::string res;
    switch (af) {
        case threshold:
            return tab + y + " = (" + x + " >= 0) ? (" + t + ")1 : (" + t + ")0;\n";
        case sym_threshold:
            return tab + y + " = (" + x + " >= 0) ? (" + t + ")1 : (" + t + ")-1;\n";
        case linear:
            res = tab + y + " = ";
            if (s != (T)1) res += "(" + t + ")" + num2str(s) + " * ";
            res += x + ";\n";
            return res;
        case sigmoid:
            res = tab + y + " = (" + t + ")1 / ((" + t + ")1 + exp(";
            if ((T)-2 * s != (T)-1) res += "(" + t + ")" + num2str((T)-2 * s) + " * ";
            else res += "-";
            res += x + "));\n";
            return res;
        case sym_sigmoid:
            res = tab + y + " = (" + t + ")2. / ((" + t + ")1. + exp(";
            if ((T)-2 * s != (T)-1) res += "(" + t + ")" + num2str((T)-2 * s) + " * ";
            else res += "-";
            res += x + ")) - (" + t + ")1.;\n";
            return res;
        case sigmoid_approx:
            res = tab + x + " = ";
            if (s != (T)1) res += "(" + t + ")" + num2str(s) + " * ";
            res += x + ";\n"
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
            return res;
        case sym_sigmoid_approx:
            res = tab + x + " = ";
            if (s != (T)1) res += "(" + t + ")" + num2str(s) + " * ";
            res += x + ";\n"
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
            return res;
        default:
            error("invalid activation function id");
    }
    return res;
}


template <typename T>
std::string
mlp_act_f_der_C_code(int af, T s, const std::string &t,
                     const std::string &x, const std::string &y,
                     const std::string &tab)
{
    std::string res;
    switch (af) {
        case linear:
            return tab + y + " = (" + t + ")" + num2str(s) + ";\n";
        case sigmoid_approx:
        case sigmoid:
            res = tab + y + " = ";
            if ((T)2 * s != (T)1) res += "(" + t + ")" + num2str((T)2 * s) + " * ";
            res += x + " * ((" + t + ")1 - " + x + ");\n";
            return res;
        case sym_sigmoid_approx:
        case sym_sigmoid:
            res = tab + y + " = ";
            if (s != (T)1) res += "(" + t + ")" + num2str(s) + " * ";
            res += "((" + t + ")1 - " + x + " * " + x + ");\n";
            return res;
        case threshold:
        case sym_threshold:
            throw exception("trying to differentiate step function");
        default:
            throw exception("invalid activation function id");
    }
    return res;
}



template <typename T>
void write_transf(std::ofstream &file, const std::string &name,
                  const T *A, const T *b, int N)
{
    std::string tname;
    int ninrow;
    if (types_eq<T, float>::val) {
        tname = "float";
        ninrow = 6;
    }
    if (types_eq<T, double>::val) {
        tname = "double";
        ninrow = 4;
    }

    file << "static const " << tname << " " << name << "_A[" << (N * N) << "] = {\n";
    file << std::setprecision(precision<T>::val);
    for (int i = 0, k = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j, ++k) {
            file << ' ' << A[j * N + i];
            if (k < (N * N - 1)) {
                file << ',';
                if (!((k + 1) % ninrow)) file << '\n';
            } else file << '\n';
        }
    }
    file << "};\n\n\n";

    file << "static const " << tname << " " << name << "_b[" << N << "] = {\n";
    file << std::setprecision(precision<T>::val);
    for (int i = 0; i < N; ++i) {
        file << ' ' << b[i];
        if (i < (N - 1)) {
            file << ',';
            if (!((i + 1) % ninrow)) file << '\n';
        } else file << '\n';
    }
    file << "};\n\n\n";

    file << "void\n" << name << "(const " << tname << " *x, "
         << tname << " *y)\n{\n";
    file << "    int k = 0, i, j;\n";
    file << "    for (i = 0; i < " << N << "; ++i) {\n"
         << "        y[i] = " << name << "_b[i];\n"
         << "        for (j = 0; j < " << N << "; ++j, ++k) {\n"
         << "            y[i] += " << name << "_A[k] * x[j];\n";
    file << "        }\n    }\n}\n\n\n";
}


} /* namespace */





template <typename T>
bool
fcnn::internal::mlp_export_C(const std::string &fname,
                             const std::string &netname,
                             const std::vector<int> &layers,
                             const std::vector<int> &n_p,
                             const std::vector<T> &w_val,
                             const std::vector<int> &w_fl,
                             int w_on,
                             const std::vector<int> &af,
                             const std::vector<T> &af_p,
                             bool with_bp,
                             const T *A, const T *b,
                             const T *C, const T *d,
                             const T *E, const T *f)
{
    std::ofstream file(fname.c_str());
    if (!file.good()) return false;

    std::string name;
    if (!netname[0]) {
        name = "export";
    } else {
        int i = 0, n = netname.length();
        for (; i < n; ++i) {
            char c = netname[i];
            if (std::isalnum(c)) name += c;
            else name += '_';
        }
    }

    bool transf_in = false, transf_out = false, transf_outinv = false;
    if (A && b) transf_in = true;
    else if (A || b) error("incomplete input transformation provided");
    if (C && d) transf_out = true;
    else if (C || d) error("incomplete output transformation provided");
    if (E && f) transf_outinv = true;
    else if (E || f) error("incomplete inverse output transformation provided");
    if (with_bp && transf_out && !transf_outinv)
        error("output transformation provided but without inverse transformation");

    file << "/*\n * This file was generated automatically by FCNN "
         << fcnn_ver() << " on " << time_str() << '\n'
         << " * Exported C function name (network evaluation): " << name << "_eval\n";
    if (with_bp) {
        file << " * Exported C function name (weights update): " << name << "_update\n";
    }
    file << " */\n\n#include <math.h>\n\n";

    std::string tname;
    int ninrow;
    if (types_eq<T, float>::val) {
        tname = "float";
        ninrow = 6;
    }
    if (types_eq<T, double>::val) {
        tname = "double";
        ninrow = 4;
    }

    file << "/* function declaration - network evaluation */\nvoid " << name << "_eval(const "
         << tname << "*, " << tname << "*);\n\n";
    if (with_bp) {
        file << "/* function declaration - weights update */\nvoid " << name << "_update(const "
             << tname << "*, const " << tname << "*);\n\n";
        file << "/* weights update parameters */\n#define LEARN_RATE 0.2\n#define MOMENTUM 0.5\n\n";
    }
    file << "\n/* ------------------------------------------------------------ */\n\n\n";

    int nol = layers.size(), non = n_p[nol], now = w_val.size();
    if (transf_in) write_transf(file, name + "_in", A, b, layers[0]);
    if (transf_out) write_transf(file, name + "_out", C, d, layers[nol - 1]);

    file << "static " << (with_bp ? "" : "const ") << tname << " "
         << name << "_w[" << now << "] = {\n";
    file << std::setprecision(precision<T>::val);
    for (int i = 0; i < now; ++i) {
        file << ' ' << w_val[i];
        if (i < (now - 1)) {
            file << ',';
            if (!((i + 1) % ninrow)) file << '\n';
        } else file << '\n';
    }
    file << "};\n\n\n";

    file << "void\n" << name << "_eval(const " << tname << " *input, "
         << tname << " *output)\n{\n";

    std::string tab = "    ";
    file << "    " << tname << " n[" << non << "];\n";
    file << "    " << tname << " x;\n";
    file << "    int k = 0, i = " << (transf_in ? n_p[1] : 0) << ", j;\n";

    file << '\n';
    if (transf_in) {
        file << tab << name << "_in(input, n);\n";
    } else {
        file << tab << "for (; i < " << n_p[1] << "; ++i) n[i] = input[i];\n";
    }
    int l = 1;
    for (; l < nol; ++l) {
        file << tab << "for (; i < " << n_p[l + 1] << "; ++i) {\n";
        file << tab << tab << "x = " << name << "_w[k++];\n";
        file << tab << tab << "for (j = "<< n_p[l - 1] << "; j < "
             << n_p[l] << "; ++j) x += n[j] * " << name << "_w[k++];\n";
        file << mlp_act_f_C_code(af[l], af_p[l], tname, "x", "n[i]", tab + tab);
        file << tab << "}\n";
    }
    --l;
    if (transf_out) {
        file << tab << name << "_out(n + " << n_p[l] << ", output);\n";
    } else {
        file << tab << "for (i = 0, j = " << n_p[l] << "; i < " << layers[l]
             << "; ++i, ++j) output[i] = n[j];\n";
    }

    file << "}\n\n\n";

    if (!with_bp) goto end;

    if (transf_outinv) write_transf(file, name + "_outinv", E, f, layers[nol - 1]);

    file << "static const int " << name << "_widx[] = {\n";
    ninrow = 10;
    for (int i = 0, j = 0; i < now; ++i) {
        if (w_fl[i]) ++j;
        file << ' ' << (w_fl[i] ? j : 0);
        if (i < (now - 1)) {
            file << ',';
            if (!((i + 1) % ninrow)) file << '\n';
        } else file << '\n';
    }
    file << "};\n\n\n";

    file << "static " << tname << " " << name << "_m[" << w_on << "] = {\n";
    ninrow = 26;
    for (int i = 0; i < w_on; ++i) {
        file << " 0";
        if (i < (w_on - 1)) {
            file << ',';
            if (!((i + 1) % ninrow)) file << '\n';
        } else file << '\n';
    }
    file << "};\n\n\n";

    file << "static " << tname << " " << name << "_gr[" << w_on << "];\n\n\n";

    file << "void\n" << name << "_update(const " << tname << " *input, const "
         << tname << " *output)\n{\n";

    file << "    " << tname << " n[" << non << "];\n";
    file << "    " << tname << " x;\n";
    file << "    " << tname << " d[" << non << "];\n";
    file << "    int k = 0, i = " << (transf_in ? n_p[1] : 0) << ", j;\n";

    file << '\n';
    if (transf_in) {
        file << tab << name << "_in(input, n);\n";
    } else {
        file << tab << "for (; i < " << n_p[1] << "; ++i) n[i] = input[i];\n";
    }
    for (l = 1; l < nol; ++l) {
        file << tab << "for (; i < " << n_p[l + 1] << "; ++i) {\n";
        file << tab << tab << "x = " << name << "_w[k++];\n";
        file << tab << tab << "for (j = "<< n_p[l - 1] << "; j < "
             << n_p[l] << "; ++j) x += n[j] * " << name << "_w[k++];\n";
        file << mlp_act_f_C_code(af[l], af_p[l], tname, "x", "n[i]", tab + tab);
        file << tab << "}\n";
    }
    --l;
    file << '\n';

    if (transf_outinv) {
        file << tab << name << "_outinv(output, d + " << n_p[l] << ");\n";
        file << tab << "for (i = " << n_p[l] << "; i < " << n_p[l + 1]
             << "; ++i) d[i] = n[i] - d[i];\n";
    } else {
        file << tab << "for (i = 0, j = " << n_p[l] << "; i < " << layers[l]
             << "; ++i, ++j) d[j] = n[j] - output[i];\n";
    }
    file << tab << "for (i = 0; i < " << n_p[l] << "; ++i, ++j) d[i] = 0;\n";
    file << tab << "for (i = 0; i < " << w_on << "; ++i) " << name << "_gr[i] = 0;\n";
    for (l = nol - 1; l > 1; --l) {
        file << tab << "for (i = " << (n_p[l + 1] - 1);
        if (l == nol - 1) file << ", --k";
        file << "; i >= " << n_p[l] << "; --i) {\n";
        file << mlp_act_f_der_C_code(af[l], af_p[l], tname, "n[i]", "x", tab + tab);
        file << tab << tab << "x *= d[i];\n";
        file << tab << tab << "for (j = " << (n_p[l] - 1) << "; j >= " << n_p[l - 1] << "; --j, --k) {\n";
        file << tab << tab << tab << "d[j] += " << name << "_w[k] * x;\n";
        file << tab << tab << tab << "if (" << name << "_widx[k]) "
             << name << "_gr[" << name << "_widx[k] - 1] += n[j] * x;\n";
        file << tab << tab << "}\n";
        file << tab << tab << "if (" << name << "_widx[k]) "
             << name << "_gr[" << name << "_widx[k] - 1] += x;\n";
        file << tab << tab << "--k;\n";
        file << tab << "}\n";
    }
    file << tab << "for (i = " << (n_p[l + 1] - 1) << "; i >= " << n_p[l] << "; --i) {\n";
    file << mlp_act_f_der_C_code(af[l], af_p[l], tname, "n[i]", "x", tab + tab);
    file << tab << tab << "x *= d[i];\n";
    file << tab << tab << "for (j = " << (n_p[l] - 1) << "; j >= " << n_p[l - 1] << "; --j, --k) {\n";
    file << tab << tab << tab << "if (" << name << "_widx[k]) "
            << name << "_gr[" << name << "_widx[k] - 1] += n[j] * x;\n";
    file << tab << tab << "}\n";
    file << tab << tab << "if (" << name << "_widx[k]) "
            << name << "_gr[" << name << "_widx[k] - 1] += x;\n";
    file << tab << tab << "--k;\n";
    file << tab << "}\n";

    file << tab << "for (i = 0; i < " << now << "; ++i) {\n";
    file << tab << tab << "k = "<< name << "_widx[i];\n";
    file << tab << tab << "if (k) {\n";
    file << tab << tab << tab << "--k;\n";
    file << tab << tab << tab << "x = "
         << "(" << tname << ")-LEARN_RATE * " << name << "_gr[k]"
         << " + (" << tname << ")MOMENTUM * " << name << "_m[k];\n";
    file << tab << tab << tab << name << "_m[k] = x;\n";
    file << tab << tab << tab << name << "_w[i] += x;\n";
    file << tab << tab << "}\n";
    file << tab << "}\n";

    file << "}\n\n";

end:
    if (!file.good()) return false;
    return true;
}



// Explicit instantiations
#ifndef FCNN_DOUBLE_ONLY
template bool fcnn::internal::mlp_export_C(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           const std::vector<int>&, int,
                                           const std::vector<int>&,
                                           const std::vector<float>&,
                                           bool,
                                           const float*, const float*,
                                           const float*, const float*,
                                           const float*, const float*);
#endif /* FCNN_DOUBLE_ONLY */
template bool fcnn::internal::mlp_export_C(const std::string&,
                                           const std::string&,
                                           const std::vector<int>&,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           const std::vector<int>&, int,
                                           const std::vector<int>&,
                                           const std::vector<double>&,
                                           bool,
                                           const double*, const double*,
                                           const double*, const double*,
                                           const double*, const double*);


