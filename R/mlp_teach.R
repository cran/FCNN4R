# #########################################################################
# This file is a part of FCNN4R.
#
# Copyright (c) Grzegorz Klima 2015
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# #########################################################################


#' Backpropagation (batch) teaching
#'
#' Backpropagation (a teaching algorithm) is a simple steepest
#' descent algorithm for MSE minimisation, in which weights are updated according
#' to (scaled) gradient of MSE.
#'
#' @note The name `backpropagation' is commonly used in two contexts, which
#' sometimes causes confusion. Firstly, backpropagation can be understood as
#' an efficient algorithm for MSE gradient computation that was first described
#' by Bryson and Ho in the '60s of 20th century and reinvented in the '80s.
#' Secondly, the name backpropagation is (more often) used to refer to the steepest
#' descent method that uses gradient of MSE computed efficiently by means
#' of the aforementioned algorithm. This ambiguity is probably caused by the fact
#' that in practically all neural network implementations, the derivatives of MSE
#' and weight updates are computed simultaneously in one backward pass (from
#' output layer to input layer).
#'
#' @param net an object of \code{mlp_net} class
#' @param input numeric matrix, each row corresponds to one input vector
#'        number of columns must be equal to the number of neurons
#'        in the network input layer
#' @param output numeric matrix with rows corresponding to expected outputs,
#'        number of columns must be equal to the number of neurons
#'        in the network output layer, number of rows must be equal to the number
#'        of input rows
#' @param tol_level numeric value, error (MSE) tolerance level
#' @param max_epochs integer value, maximal number of epochs (iterations)
#' @param report_freq integer value, progress report frequency, if set to 0 no information is printed
#'        on the console (this is the default)
#' @param learn_rate numeric value, learning rate in the backpropagation
#'        algorithm (default 0.7)
#'
#' @return Two-element list, the first field (\code{net}) contains trained network,
#'         the second (\code{mse}) - the learning history (MSE in consecutive epochs).
#'
#' @references
#' A.E. Bryson and Y.C. Ho. \emph{Applied optimal control: optimization, estimation,
#' and control. Blaisdell book in the pure and applied sciences.} Blaisdell Pub. Co., 1969.
#'
#' David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. \emph{Learning representations
#' by back-propagating errors.} Nature, 323(6088):533-536, October 1986.
#'
#'
#' @keywords teaching
#'
#' @export mlp_teach_bp
#'
mlp_teach_bp <- function(net, input, output,
                         tol_level, max_epochs, report_freq = 0,
                         learn_rate = 0.7)
{
    if (tol_level <= 0) stop("tolerance level should be positive")

    gm <- mlp_grad(net, input, output)
    g <- gm$grad
    mse <- gm$mse
    if (mse < tol_level) {
        return(list(net = net, mse = NULL))
    }

    mseh <- numeric(length = max_epochs)
    for (i in 1:max_epochs) {
        w0 <- mlp_get_weights(net)
        w1 <- w0 - learn_rate * g
        net <- mlp_set_weights(net, w1)
        gm <- mlp_grad(net, input, output)
        g <- gm$grad
        mse <- gm$mse
        mseh[i] <- mse
        if (report_freq) {
            if (!(i %% report_freq)) {
                mes <- paste0("backpropagation; epoch ", i,
                              ", mse: ", mse, " (desired: ", tol_level, ")\n")
                cat(mes);
            }
        }
        if (mse < tol_level) break;
    }
    if (mse > tol_level) {
        warning(paste0("algorithm did not converge, mse after ", i,
                       " epochs is ", mse, " (desired: ", tol_level, ")"))
    }
    return(list(net = net, mse = mseh[1:i]))
}




#' Rprop teaching
#'
#' Rprop is a fast and robust adaptive step method based on backpropagation.
#' For details please refer to the original paper given in References section.
#'
#'
#' @param net an object of \code{mlp_net} class
#' @param input numeric matrix, each row corresponds to one input vector
#'        number of columns must be equal to the number of neurons
#'        in the network input layer
#' @param output numeric matrix with rows corresponding to expected outputs,
#'        number of columns must be equal to the number of neurons
#'        in the network output layer, number of rows must be equal to the number
#'        of input rows
#' @param tol_level numeric value, error (MSE) tolerance level
#' @param max_epochs integer value, maximal number of epochs (iterations)
#' @param report_freq integer value, progress report frequency, if set to 0 no information is printed
#'        on the console (this is the default)
#' @param u numeric value, Rprop algorithm parameter (default 1.2)
#' @param d numeric value, Rprop algorithm parameter (default 0.5)
#' @param gmax numeric value, Rprop algorithm parameter (default 50)
#' @param gmin numeric value, Rprop algorithm parameter (default 1e-6)
#'
#' @return Two-element list, the first field (\code{net}) contains trained network,
#'         the second (\code{mse}) - the learning history (MSE in consecutive epochs).
#'
#' @references
#' M. Riedmiller. \emph{Rprop - Description and Implementation Details: Technical Report.} Inst. f.
#' Logik, Komplexitat u. Deduktionssysteme, 1994.
#'
#' @keywords teaching
#'
#' @export mlp_teach_rprop
#'
mlp_teach_rprop <- function(net, input, output,
                            tol_level, max_epochs, report_freq = 0,
                            u = 1.2, d = 0.5, gmax = 50., gmin = 1e-6)
{
    if (tol_level <= 0) stop("tolerance level should be positive")

    gm <- mlp_grad(net, input, output)
    g0 <- gm$grad
    mse <- gm$mse
    if (mse < tol_level) {
        return(list(net = net, mse = NULL))
    }
    w0 <- mlp_get_weights(net);
    w1 <- w0 - 0.7 * g0;
    net <- mlp_set_weights(net, w1)

    mseh <- numeric(length = max_epochs)
    gm <- mlp_grad(net, input, output)
    g1 <- gm$grad
    mse <- gm$mse
    mseh[1] <- mse
    if (report_freq == 1) {
        mes <- paste0("Rprop; epoch 1",
                        ", mse: ", mse, " (desired: ", tol_level, ")\n")
        cat(mes);
    }
    if (mse < tol_level) {
        return(list(net = net, mse = mse))
    }

    nw <- length(w0)
    if (gmin > 1e-1) {
        gam <- gmin
    } else {
        gam <- min(0.1, gmax)
    }
    gamma <- rep(gam, nw)

    for (i in 2:max_epochs) {
        # determine step and update gamma
        dw <- rep(0, nw)
        ig0 <- (g1 > 0)
        il0 <- (g1 < 0)
        i1 <- (g0 * g1 > 0)
        ind <- which(i1 & ig0)
        dw[ind] <- -gamma[ind]
        ind <- which(i1 & !ig0)
        dw[ind] <- gamma[ind]
        gamma[i1] <- pmin(u * gamma[i1], gmax)
        i2 <- (g0 * g1 < 0)
        ind <- which(i2)
        dw[ind] <- 0
        gamma[ind] <- pmax(d * gamma[ind], gmin)
        i3 <- (g0 * g1 == 0)
        ind <- which(i3 & ig0)
        dw[ind] <- -gamma[ind]
        ind <- which(i3 & il0)
        dw[ind] <- gamma[ind]
        # update weights
        w0 <- mlp_get_weights(net);
        w1 <- w0 + dw;
        net <- mlp_set_weights(net, w1)
        # new gradients
        g0 <- g1;
        gm <- mlp_grad(net, input, output)
        g1 <- gm$grad
        mse <- gm$mse
        mseh[i] <- mse
        if (report_freq) {
            if (!(i %% report_freq)) {
                mes <- paste0("Rprop; epoch ", i, ", mse: ", mse,
                              " (desired: ", tol_level, ")\n")
                cat(mes);
            }
        }
        if (mse < tol_level) break;
    }
    if (mse > tol_level) {
        warning(paste0("algorithm did not converge, mse after ", i,
                       " epochs is ", mse, " (desired: ", tol_level, ")"))
    }
    return(list(net = net, mse = mseh[1:i]))
}
