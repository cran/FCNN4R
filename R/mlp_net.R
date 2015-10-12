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

.onAttach <- function( ... )
{
  packageStartupMessage(paste0("Fast Compressed Neural Networks for R ",
                        utils::packageVersion("FCNN4R"),
                        "\nhttp://fcnn.sourceforge.net/"))
}

#' Fast Compressed Neural Networks for R
#'
#' The FCNN4R package provides an interface to kernel routines
#' from the FCNN C++ library. FCNN is based on a completely new
#' Artificial Neural Network representation that offers unmatched
#' efficiency, modularity, and extensibility. FCNN4R provides
#' standard teaching (backpropagation, Rprop, simulated annealing)
#' and pruning algorithms (minimum magnitude, Optimal Brain Surgeon),
#' but it is first and foremost an efficient computational engine.
#' Users can easily implement their algorithms by taking advantage
#' of fast gradient computing routines, as well as network
#' reconstruction functionality (removing weights and redundant
#' neurons).
#'
#' @name FCNN4R-package
#'
#' @author Grzegorz Klima <gklima@@users.sourceforge.net>
#'
#' @references
#' G. Klima. \emph{A new approach towards implementing artificial neural networks.}
#' Technical Report, \url{http://fcnn.sourceforge.net/}, 2013.
#'
#' @keywords package
#'
#' @examples
#'
#' # create a 2-6-1 network
#' net <- mlp_net(c(2, 6, 1))
#' # set up the XOR problem inputs and outputs
#' inp <- c(0, 0, 1, 1, 0, 1, 0, 1)
#' dim(inp) <- c(4, 2)
#' outp <- c(0, 1, 1, 0)
#' dim(outp) <- c(4, 1)
#' # randomise weights
#' net <- mlp_rnd_weights(net)
#' # tolerance level
#' tol <- 0.5e-4
#' # teach using Rprop, assign trained network and plot learning history
#' netmse <- mlp_teach_rprop(net, inp, outp, tol_level = tol,
#'                           max_epochs = 500, report_freq = 10)
#' net <- netmse$net
#' # if the algorithm did not converge, teach again with new random initial weights
#' while (mlp_mse(net, inp, outp) > tol) {
#'     net <- mlp_rnd_weights(net)
#'     netmse <- mlp_teach_rprop(net, inp, outp, tol_level = tol,
#'                             max_epochs = 500, report_freq = 10)
#'     net <- netmse$net
#' }
#' # plot learning history
#' plot(netmse$mse, type = 'l')
#' # plot network with weights
#' mlp_plot(net, TRUE)
#' # prune using Optimal Brain Surgeon
#' net <- mlp_prune_obs(net, inp, outp, tol_level = 0.5e-4,
#'                      max_reteach_epochs = 500, report = TRUE)[[1]]
#' # plot network with weights
#' mlp_plot(net, TRUE)
#' # check network output
#' round(mlp_eval(net, inp), digits = 3)
#'
#' @useDynLib FCNN4R
#'
#' @import methods
#' @import Rcpp
#' @importFrom stats runif rnorm
#' @importFrom graphics plot.new plot.window segments text points
#'
NULL

#' An S4 class representing Multilayer Perception Network.
#'
#' The \code{mlp_net} class represents the Multilayer Perception Network
#' employing the so-called compressed representation, which was inspired
#' by the Compressed Column Storage familiar from sparse matrix algebra.
#' Although the representation and algorithms working with it are somewhat
#' complicated, the user is provided with a simple and intuitive interface
#' that completely hides the internal workings of the package, which in its
#' large part is written in C++.
#'
#' @aliases mlp_net-class mlp_net-method show,mlp_net-method summary,mlp_net-method
#'
#' @slot m_name character string, network name
#' @slot m_layers integer vector, stores the numbers of neurons in layers
#' @slot m_n_pointers integer vector, stores the so-called 'pointers' to neurons
#' @slot m_n_prev integer vector, stores the number of connected neurons in the previous layer
#' @slot m_n_next integer vector, stores the number of connected neurons in the next layer
#' @slot m_w_pointers integer vector, stores the so-called 'pointers' to weights
#' @slot m_w_values numeric vector, values of connection weights and biases
#' @slot m_w_flags logical vector, states (active/inactive) of weights and biases
#' @slot m_w_on integer value, the number of active weights
#' @slot m_af_hl integer value, hidden layer activation function index
#' @slot m_af_hl_slope numeric value, hidden layer activation function slope parameter
#' @slot m_af_ol integer value, output layer activation function index
#' @slot m_af_ol_slope numeric value, output layer activation function slope parameter
#'
#' @seealso \code{\link{mlp_net}} for creating objects of this class.
#'
#' @references
#' G. Klima. \emph{A new approach towards implementing artificial neural networks.}
#' Technical Report, \url{http://fcnn.sourceforge.net/}, 2013.
#'
#' @keywords classes
#'
#' @name mlp_net-class
#'
setClass(Class = "mlp_net",
    representation(
        m_name = "character",
        m_layers = "integer",
        m_n_pointers = "integer",
        m_n_prev = "integer",
        m_n_next = "integer",
        m_w_pointers = "integer",
        m_w_values = "numeric",
        m_w_flags = "integer",
        m_w_on = "integer",
        m_af_hl = "integer",
        m_af_hl_slope = "numeric",
        m_af_ol = "integer",
        m_af_ol_slope = "numeric"
    ),
    package = "FCNN4R"
)

#' Create objects of \code{mlp_net} class
#'
#' Function used for creating multilayer perceptron networks.
#'
#' @param layers vector providing numbers of neurons in each layer
#' @param name character string, network name (optional)
#'
#' @return Returns an object of \code{mlp_net} class.
#'
#' @seealso \code{\linkS4class{mlp_net}} for details.
#'
#' @examples
#'
#' # create a 2-3-1 network
#' net <- mlp_net(c(2, 3, 1))
#' # randomise weights
#' net <- mlp_rnd_weights(net)
#' # show basic information about the network
#' show(net)
#'
#' @keywords classes
#'
#' @export mlp_net
#'
mlp_net <- function(layers, name = NULL)
{
    layers <- as.integer(layers)
    if (is.null(name)) {
        name <- ""
    } else {
        if (!is.character(name) || (length(name) != 1)) {
            stop("invalid network name")
        }
    }
    cres <- .Call("mlp_construct", layers)
    object <- new("mlp_net",
                  m_name = name,
                  m_layers = layers,
                  m_n_pointers = cres[[1]],
                  m_n_prev = cres[[2]],
                  m_n_next = cres[[3]],
                  m_w_pointers = cres[[4]],
                  m_w_values = cres[[5]],
                  m_w_flags = cres[[6]],
                  m_w_on = cres[[7]],
                  m_af_hl = 4L,
                  m_af_hl_slope = .5,
                  m_af_ol = 4L,
                  m_af_ol_slope = .5)
    return (object)
}


#' Is it an object of \code{mlp_net} class?
#'
#' This function checks whether argument is an object of \code{mlp_net} class.
#'
#' @param x an object to be checked
#'
#' @return Logical value.
#'
#' @keywords classes
#'
#' @export is.mlp_net
#'
is.mlp_net <- function(x)
{
    if (is(x, "mlp_net")) return(TRUE)
    return(FALSE)
}



#' @export
#'
setMethod("show", signature(object = "mlp_net"),
function(object)
{
    if (object@m_name != "") {
        cat(paste0("Multilayer perceptron network (", object@m_name, ")\n"))
    } else {
        cat("Multilayer perceptron network\n")
    }
    lays <- object@m_layers
    nlays <- length(lays)
    cat(paste0("Layers: ", lays[1], "(input) - ",
        paste0(lays[2:(nlays - 1)], collapse = " - "),
        " - ", lays[nlays], "(output)\n"))
    cat(paste0("Active weights (connections & biases): ",
               object@m_w_on, " of ", object@m_w_pointers[nlays + 1], "\n"))
    cat("Activation functions:\n")
    cat(paste0("  hidden layer: ",
        mlp_actvfunc2str(object@m_af_hl, object@m_af_hl_slope), "\n"))
    cat(paste0("  output layer: ",
        mlp_actvfunc2str(object@m_af_ol, object@m_af_ol_slope), "\n"))
    cat("Weights:\n")
    now <- length(object@m_w_flags)
    truncthresh <- 19
    if (now > truncthresh) {
        trunc <- TRUE
        wg <- as.character(object@m_w_values[1:truncthresh])
        wg[which(object@m_w_flags[1:truncthresh] == 0L)] <- "off"
        wg <- c(wg, "...[truncated]")
    } else {
        trunc <- FALSE
        wg <- as.character(object@m_w_values)
        wg[which(object@m_w_flags == 0L)] <- "off"
    }
    cat(wg)
})




#' @export
#'
setMethod("summary", signature(object = "mlp_net"),
function(object)
{
    if (object@m_name != "") {
        cat(paste0("Multilayer perceptron network (", object@m_name, ")\n"))
    } else {
        cat("Multilayer perceptron network\n")
    }
    lays <- object@m_layers
    nlays <- length(lays)
    cat(paste0("Layers: ", lays[1], "(input) - ",
        paste0(lays[2:(nlays - 1)], collapse = " - "),
        " - ", lays[nlays], "(output)\n"))
    cat(paste0("Active weights (connections & biases): ",
               object@m_w_on, " of ", object@m_w_pointers[nlays + 1], "\n"))
    cat("Activation functions:\n")
    cat(paste0("  hidden layer: ",
        mlp_actvfunc2str(object@m_af_hl, object@m_af_hl_slope), "\n"))
    cat(paste0("  output layer: ",
        mlp_actvfunc2str(object@m_af_ol, object@m_af_ol_slope), "\n"))
    cat("Weights:\n")
    for (l in 2:(nlays)) {
        if (l < nlays) {
            cat(paste0("  layer ", l, " (hidden layer ", l - 1, "): \n"))
        } else {
            cat(paste0("  layer ", nlays, " (output layer): \n"))
        }
        for (n in (1:lays[l])) {
            cat(paste0("    neuron ", n, ":\n"))
            cat("      bias: ")
            if (!mlp_get_w_st(object, layer = l, nidx = n, nplidx = 0)) {
                cat("off\n")
            } else {
                cat(paste0(mlp_get_w(object, layer = l, nidx = n, nplidx = 0), "\n"))
            }
            for (np in (1:lays[l - 1])) {
                cat(paste0("      conn. to neuron ", np, " in layer ", l - 1, ": "))
                if (!mlp_get_w_st(object, layer = l, nidx = n, nplidx = np)) {
                    cat("off\n")
                } else {
                    cat(paste0(mlp_get_w(object, layer = l, nidx = n, nplidx = np), "\n"))
                }
            }
        }
    }
})




# #########################################################################
# Network names
# #########################################################################


#' Get and set network names
#'
#' The following functions can be used for retrieving and setting network names.
#'
#' @param net an object of \code{mlp_net} class
#' @param name character string with network name
#'
#' @return \code{mlp_get_name} returns character string with network name.
#'
#'         \code{mlp_set_name} returns network (an object of \code{mlp_net}
#'               class) with name set to new value.
#'
#' @name mlp_net-names
#'
#' @export
#'
mlp_get_name <- function(net)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    return(net@m_name)
}



#' @rdname mlp_net-names
#'
#' @export
#'
mlp_set_name <- function(net, name)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (!is.character(name) || (length(name) != 1)) {
        stop("invalid network name")
    }
    net@m_name <- name
    return(net)
}



# #########################################################################
# General information about network
# #########################################################################


#' General information about network
#'
#' The following functions return basic information about the network.
#'
#' @param net an object of \code{mlp_net} class
#'
#' @return \code{mlp_get_layers} returns an integer vector with numbers of neurons in consecutive layers.
#'
#'         \code{mlp_get_no_active_w} returns the number of active weights (connections and biases).
#'
#'         \code{mlp_get_no_w} returns the total number (including inactive) of weights
#'                    (connections and biases).
#'
#' @seealso \code{\link[=mlp_net-class]{mlp_net-class}} for details
#'          on internal network representation.
#'
#' @name mlp_net-general-information
#'
#' @export
#'
mlp_get_layers <- function(net)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    return(net@m_layers)
}

#' @rdname mlp_net-general-information
#'
#' @export
#'
mlp_get_no_active_w <- function(net)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    return(net@m_w_on)
}


#' @rdname mlp_net-general-information
#'
#' @export
#'
mlp_get_no_w <- function(net)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    return(length(net@m_w_flags))
}



# #########################################################################
# Reconstructing network, removing neurons
# #########################################################################

#' Remove redundant neurons in a multilayer perceptron network
#'
#' This function removes redundant neurons from the network, i.e. hidden layers'
#' neurons that are not connected to neurons in the previous layer or the next
#' layer. If a neuron is not connected to neurons in the previous layer but
#' is connected to neurons in the next layer (effectively acts as an additional
#' bias), biases of neurons in the next layer are properly adjusted, therefore
#' the resulting network behaves just like the initial one.
#'
#' @param net an object of \code{mlp_net} class
#' @param report logical value, if TRUE, information about removed neurons
#'        will be printed on the console (FALSE by default)
#'
#' @return Three-element list. The first element (\code{net}) is the network
#'         (an object of \code{mlp_net} class) with all redundant neurons
#'         removed, the second (\code{ncount}) - the number of neurons removed,
#'         the third (\code{wcount}) - the number of weights removed.
#'
#' @export mlp_rm_neurons
#'
mlp_rm_neurons <- function(net, report = FALSE)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    hst <- net@m_n_pointers[2] + 1
    hen <- net@m_n_pointers[length(net@m_layers)]
    if (all(net@m_n_prev[hst:hen] != 0) && all(net@m_n_next[hst:hen] != 0)) {
        return(list(net = net, ncount = 0, wcount = 0))
    }
    won0 <- net@m_w_on
    rmres <- .Call("mlp_rm_neurons",
                   net@m_layers,
                   net@m_n_pointers,
                   net@m_n_prev,
                   net@m_n_next,
                   net@m_w_pointers,
                   net@m_w_values,
                   net@m_w_flags,
                   net@m_w_on,
                   net@m_af_hl,
                   net@m_af_hl_slope,
                   report)
    ret <- new("mlp_net",
               m_name = net@m_name,
               m_layers = rmres[[1]],
               m_n_pointers = rmres[[2]],
               m_n_prev = rmres[[3]],
               m_n_next = rmres[[4]],
               m_w_pointers = rmres[[5]],
               m_w_values = rmres[[6]],
               m_w_flags = rmres[[7]],
               m_w_on = rmres[[8]],
               m_af_hl = net@m_af_hl,
               m_af_hl_slope = net@m_af_hl_slope,
               m_af_ol = net@m_af_ol,
               m_af_ol_slope = net@m_af_ol_slope)
    return(list(net = ret, ncount = rmres[[9]], wcount = won0 - rmres[[8]]))
}




# #########################################################################
# Combining two networks into one
# #########################################################################

#' Combining two networks into one
#'
#' These functions construct new network by merging two networks
#' (they must have the same number of layers) or by connecting
#' one network outputs to another network inputs (the numbers of output
#' and input neurons must agree). These functions may be used in constructing
#' deep learning networks or constructing networks with some special topologies.
#'
#' @param net1 an object of \code{mlp_net} class
#' @param net2 an object of \code{mlp_net} class
#' @param same_inputs logical, if TRUE both merged networks are assumed to take
#'          the same inputs (they share the input layer), default is FALSE
#'
#' @return Both functions return an object of \code{mlp_net} class.
#'
#' @examples
#'
#' # create two 2-2-2 networks with random weights and plot them
#' net1 <- mlp_net(c(2, 2, 2))
#' net1 <- mlp_rnd_weights(net1)
#' mlp_plot(net1, TRUE)
#' net2 <- mlp_net(c(2, 2, 2))
#' net2 <- mlp_rnd_weights(net2)
#' mlp_plot(net2, TRUE)
#' # create a 4-3-2 network with random weights and plot it
#' net3 <- mlp_net(c(4, 3, 2))
#' net3 <- mlp_rnd_weights(net3)
#' mlp_plot(net3, TRUE)
#' # construct new network using net1, net2, and net3 and plot it
#' net4 <- mlp_stack(mlp_merge(net1, net2), net3)
#' mlp_plot(net4, TRUE)
#'
#' @name mlp_net-combining-two-networks
#'
#' @export
#'
mlp_merge <- function(net1, net2, same_inputs = FALSE)
{
    if (!is.mlp_net(net1)) {
        stop("expected net1 argument to be of mlp_net class")
    }
    if (!is.mlp_net(net2)) {
        stop("expected net2 argument to be of mlp_net class")
    }
    if (!is.logical(same_inputs)) {
        stop("expected logical argument")
    }
    if ((net1@m_af_hl != net2@m_af_hl) || (net1@m_af_hl_slope != net2@m_af_hl_slope)
        || (net1@m_af_ol != net2@m_af_ol) || (net1@m_af_ol_slope != net2@m_af_ol_slope)) {
        stop("activation functions in networks disagree");
    }
    res <- .Call("mlp_merge",
                 net1@m_layers,
                 net1@m_w_pointers,
                 net1@m_w_values,
                 net1@m_w_flags,
                 net2@m_layers,
                 net2@m_w_pointers,
                 net2@m_w_values,
                 net2@m_w_flags,
                 same_inputs)
    net <- new("mlp_net",
               m_name = "",
               m_layers = res[[1]],
               m_n_pointers = res[[2]],
               m_n_prev = res[[3]],
               m_n_next = res[[4]],
               m_w_pointers = res[[5]],
               m_w_values = res[[6]],
               m_w_flags = res[[7]],
               m_w_on = res[[8]],
               m_af_hl = net1@m_af_hl,
               m_af_hl_slope = net1@m_af_hl_slope,
               m_af_ol = net1@m_af_ol,
               m_af_ol_slope = net1@m_af_ol_slope)
    return(net)
}


#' @rdname mlp_net-combining-two-networks
#'
#' @export
#'
mlp_stack <- function(net1, net2)
{
    if (!is.mlp_net(net1)) {
        stop("expected net1 argument to be of mlp_net class")
    }
    if (!is.mlp_net(net2)) {
        stop("expected net2 argument to be of mlp_net class")
    }
    if ((net1@m_af_hl != net1@m_af_ol) || (net1@m_af_hl_slope != net1@m_af_ol_slope)) {
        stop("the 1st network must have the same activation functions in both hidden and output layers")
    }
    if ((net1@m_af_ol != net2@m_af_hl) || (net1@m_af_ol_slope != net2@m_af_hl_slope)) {
        stop("the 1st network's activation function must agree with the 2nd network's activation function in the hidden layer(s)")
    }
    res <- .Call("mlp_stack",
                 net1@m_layers,
                 net1@m_w_pointers,
                 net1@m_w_values,
                 net1@m_w_flags,
                 net2@m_layers,
                 net2@m_w_pointers,
                 net2@m_w_values,
                 net2@m_w_flags)
    net <- new("mlp_net",
               m_name = "",
               m_layers = res[[1]],
               m_n_pointers = res[[2]],
               m_n_prev = res[[3]],
               m_n_next = res[[4]],
               m_w_pointers = res[[5]],
               m_w_values = res[[6]],
               m_w_flags = res[[7]],
               m_w_on = res[[8]],
               m_af_hl = net1@m_af_hl,
               m_af_hl_slope = net1@m_af_hl_slope,
               m_af_ol = net1@m_af_ol,
               m_af_ol_slope = net1@m_af_ol_slope)
    return(net)
}



# #########################################################################
# Importing and exporting networks
# #########################################################################

#' Export and import multilayer perceptron network to/from a text file
#' in FCNN format
#'
#' These functions can be used to export and import multilayer perceptron
#' network to/from a text file in FCNN format.
#'
#' Files are organised as follows:
#' \itemize{
#'  \item the first comment (beginning with \code{#}) is treated as network information (name) string,
#'  \item all other comments are ignored,
#'  \item network structure is represented by five lines of numbers:
#'     \itemize{
#'      \item the first line determines numbers of neurons in consecutive layers,
#'      \item the second line of 0's and 1's determines which weights are turned off/on,
#'      \item the third line contains active weights' values,
#'      \item the last two lines determine hidden and output layer activation functions
#'            and their slope parameters.
#'      }
#'  }
#'
#' @param net an object of \code{mlp_net} class
#' @param fname character string with the filename
#'
#' @return \code{mlp_export_fcnn} returns logical value, TRUE if export was successful, FALSE otherwise.
#'
#'         \code{mlp_ipport_fcnn} returns  an object of \code{mlp_net} class or NULL, if import failed.
#'
#' @seealso \code{\linkS4class{mlp_net}} for network representation details.
#'
#' @examples
#'
#' # create a 2-3-1 network
#' net <- mlp_net(c(2, 3, 1))
#' # randomise weights
#' net <- mlp_rnd_weights(net)
#' # Show the network
#' show(net)
#' # export network
#' mlp_export_fcnn(net, "test.net")
#' # Show the output file
#' file.show("test.net")
#' # import network
#' net2 <- mlp_import_fcnn("test.net")
#' # Show the imported network
#' show(net2)
#'
#' @name mlp_net-export-import
#'
#' @export
#'
mlp_export_fcnn <- function(net, fname)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (!is.character(fname) || (length(fname) != 1)) {
        stop("invalid filename")
    }
    return(.Call("mlp_export", fname,
                 net@m_name, net@m_layers, net@m_w_values, net@m_w_flags,
                 net@m_af_hl, net@m_af_hl_slope, net@m_af_ol, net@m_af_ol_slope))
}

#' @rdname mlp_net-export-import
#'
#' @export
#'
mlp_import_fcnn <- function(fname)
{
    if (!is.character(fname) || (length(fname) != 1)) {
        stop("invalid filename")
    }
    impres <- .Call("mlp_import", fname)
    if (is.null(impres)) {
        return(impres)
    }
    object <- new("mlp_net",
                  m_name = impres[[1]],
                  m_layers = impres[[2]],
                  m_n_pointers = impres[[3]],
                  m_n_prev = impres[[4]],
                  m_n_next = impres[[5]],
                  m_w_pointers = impres[[6]],
                  m_w_values = impres[[7]],
                  m_w_flags = impres[[8]],
                  m_w_on = impres[[9]],
                  m_af_hl = impres[[10]],
                  m_af_hl_slope = impres[[11]],
                  m_af_ol = impres[[12]],
                  m_af_ol_slope = impres[[13]])
    return(object)
}



# #########################################################################
# Exporting networks to C
# #########################################################################

#' Export multilayer perceptron network to a C function
#'
#' This function exports multilayer perceptron network to a C function
#' with optional affine input and output transformations: Ax+b for inputs
#' and Cx+d for outputs.
#'
#' @param net an object of \code{mlp_net} class
#' @param fname character string with the filename
#' @param A numeric matrix (optional), input linear transformation
#' @param b numeric vector (optional), input translation
#' @param C numeric matrix (optional), output linear transformation
#' @param d numeric vector (optional), output translation
#'
#' @return Logical value, TRUE if export was successful, FALSE otherwise.
#'
#' @examples
#'
#' # create a 2-3-1 network
#' net <- mlp_net(c(2, 3, 1))
#' # randomise weights
#' net <- mlp_rnd_weights(net)
#' # show the network
#' show(net)
#' # export network to a C function
#' mlp_export_C(net, "test.c")
#' # show the output file
#' file.show("test.c")
#'
#' @export mlp_export_C
#'
mlp_export_C <- function(net, fname, A = NULL, b = NULL, C = NULL, d = NULL)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (!is.character(fname) || (length(fname) != 1)) {
        stop("invalid filename")
    }
    if (is.null(A)) {
        A <- diag(net@m_layers[1])
    } else {
        if (!is.numeric(A) || (length(dim(A)) != 2)) {
            stop("A must be a numeric matrix")
        }
        if (!all(dim(A) == net@m_layers[1]))
            stop("invalid sizes of matrix A")
    }
    if (is.null(b)) {
        b <- rep(0, length = net@m_layers[1])
    } else {
        if (!is.numeric(b)) {
            stop("b must be a numeric vector")
        }
        if (length(b) != net@m_layers[1])
            stop("invalid size of vector b")
    }
    if (is.null(C)) {
        C <- diag(net@m_layers[length(net@m_layers)])
    } else {
        if (!is.numeric(C) || (length(dim(C)) != 2)) {
            stop("C must be a numeric matrix")
        }
        if (!all(dim(C) == net@m_layers[length(net@m_layers)]))
            stop("invalid sizes of matrix C")
    }
    if (is.null(d)) {
        d <- rep(0, length = net@m_layers[length(net@m_layers)])
    } else {
        if (!is.numeric(d)) {
            stop("d must be a numeric vector")
        }
        if (length(d) != net@m_layers[length(net@m_layers)])
            stop("invalid size of vector d")
    }

    return(.Call("mlp_export_C", fname,
                 net@m_name, net@m_layers, net@m_n_pointers, net@m_w_values,
                 net@m_af_hl, net@m_af_hl_slope, net@m_af_ol, net@m_af_ol_slope,
                 A, b, C, d))
}



# #########################################################################
# Activation functions
# #########################################################################


#' Return character string representing activation function
#'
#' @param idx activation function index
#' @param slope activation function index slope parameter
#'
#' @return This function returns character string representing activation function.
#'
#' @keywords internal
#'
mlp_actvfunc2str <- function(idx, slope)
{
    strng <- .Call("actvfuncstr", idx)
    if (idx > 2) strng <- paste0(strng, " with s = ", slope)
    return (strng)
}



#' Set network activation functions
#'
#' This function sets activation function (and its slope parameter)
#' for neurons in the hidden layers and in the output layer.
#'
#' @param net an object of \code{mlp_net} class
#' @param output logical value, if TRUE set activation function in the output layer,
#'               if FALSE set activation function in the hidden layer(s)
#' @param activation character string, activation function name, admissible
#'                   options are: "threshold", "sym_threshold", "linear",
#'                   "sigmoid", "sym_sigmoid", "sigmoid_approx", "sym_sigmoid_approx"
#' @param slope numeric value, activation function slope parameter
#'
#' @return This function returns network (an object of \code{mlp_net} class)
#'         with activation function set.
#'
#' @export mlp_set_activation
#'
mlp_set_activation <- function(net,
                               output = FALSE,
                               activation = c("threshold", "sym_threshold", "linear",
                                              "sigmoid", "sym_sigmoid",
                                              "sigmoid_approx", "sym_sigmoid_approx"),
                               slope = NULL)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    afi <- switch(EXPR = activation, threshold = 1, sym_threshold = 2, linear = 3,
                                     sigmoid = 4, sym_sigmoid = 5,
                                     sigmoid_approx = 6, sym_sigmoid_approx = 7)
    if (is.null(afi)) {
        stop("invalid activation function name")
    }
    afi <- as.integer(afi)
    if ((afi %in% c(1, 2)) && !is.null(slope)) {
        stop("provided slope parameter for threshold function")
    }
    if (!is.null(slope) && (!is.numeric(slope) || !is.finite(slope) || (length(slope) != 1))) {
        stop("invalid slope parameter")
    }
    if (output) {
        net@m_af_ol <- afi
        if (!is.null(slope)) {
            net@m_af_ol_slope <- slope
        }
    } else {
        net@m_af_hl <- afi
        if (!is.null(slope)) {
            net@m_af_hl_slope <- slope
        }
    }
    return(net)
}



# #########################################################################
# Weights indexing
# #########################################################################


#' Check validity of weight index
#'
#' @param net an object of \code{mlp_net} class
#' @param idx integer value, weight absolute index
#' @param layer integer value, layer index
#' @param nidx integer value, neuron index
#' @param nplidx integer value, index of the neuron in the previous layer determining connection
#'               from neuron \code{nidx} in \code{layer}, 0 denotes bias
#'               of neuron \code{nidx} in \code{layer}
#'
#' @return This function does not return.
#'
#' @keywords internal
#'
mlp_check_w <- function(net, idx = NULL, layer = NULL, nidx = NULL, nplidx = NULL)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (!is.null(idx)) {
        if ((idx < 1) || (idx > length(net@m_w_values))) {
            stop("invalid weight index")
        }
        if (!is.null(layer) || !is.null(nidx) || !is.null(nplidx)) {
            stop("weight idx already provided, other arguments should not be set")
        }
    } else {
        if (is.null(layer) || is.null(nidx) || is.null(nplidx)) {
            stop("weight idx not given, 3 arguments (layer, nidx, nplidx) required")
        }
        if ((layer < 2) || (layer > length(net@m_layers))) {
            stop("invalid layer")
        }
        if ((nidx < 1) || (nidx > net@m_layers[layer])) {
            stop("invalid neuron index (nidx)")
        }
        if ((nplidx < 0) || (nplidx > net@m_layers[layer - 1])) {
            stop("invalid previous layer neuron index (nplidx)")
        }
    }
}


#' Retrieving absolute weight index
#'
#' In some situations weight index (absolute, i.e. within all weights including
#' inactive ones) needs to be computed based on information about connected
#' neurons' indices or weight index within actives ones. The latter functionality
#' is especially useful in implementation of pruning algorithms.
#'
#' @param net an object of \code{mlp_net} class
#' @param layer integer value, layer index
#' @param nidx integer value, neuron index
#' @param nplidx integer value, index of the neuron in the previous layer determining connection
#'               from neuron \code{nidx} in \code{layer}, 0 denotes bias
#'               of neuron \code{nidx} in \code{layer}
#' @param idx integer value, weight index withing active ones
#'
#' @return Absolute weight index.
#'
#' @name mlp_net-weight-indexing
#'
#' @export
#'
mlp_get_w_idx <- function(net, layer, nidx, nplidx)
{
    mlp_check_w(net, idx = NULL, layer = layer, nidx = nidx, nplidx = nplidx)
    idx <- net@m_w_pointers[layer] +
           (nidx - 1) * (net@m_layers[layer - 1] + 1) + nplidx + 1
    return(idx)
}


#' @name mlp_net-weight-indexing
#'
#' @export
#'
mlp_get_w_abs_idx <- function(net, idx)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if ((idx < 1) || (idx > net@m_w_on)) {
        stop("active weight index outside bounds")
    }
    return(.Call("mlp_get_abs_w_idx", net@m_w_flags, as.integer(idx)))
}



# #########################################################################
# Single weight access
# #########################################################################

#' Setting and retrieving single weight's status (on/off) and value
#'
#' The following functions can be used to access a single weight, i.e. set or
#' retrieve its status (on/off) and value.
#'
#' @param net an object of \code{mlp_net} class
#' @param on logical value, should the weight be set on or off?
#' @param val numeric value, connection (or bias) value to be set
#' @param idx integer value, weight absolute index
#' @param layer integer value, layer index
#' @param nidx integer value, neuron index
#' @param nplidx integer value, index of the neuron in the previous layer determining connection
#'               from neuron \code{nidx} in \code{layer}, 0 denotes bias
#'               of neuron \code{nidx} in \code{layer}
#'
#' @return \code{mlp_set_w_st} returns network (an object of \code{mlp_net} class)
#'                    with state (on/off) of selected  weight set.
#'
#'         \code{mlp_set_w} returns network (an object of \code{mlp_net} class)
#'                    with value of selected weight set.
#'
#'         \code{mlp_get_w_st} returns logical value, TRUE if connection/bias is active,
#'                    FALSE otherwise.
#'
#'         \code{mlp_get_w} returns numeric value, selected weight value.
#'
#' @name mlp_net-single-weight-access
#'
#' @export
#'
mlp_set_w_st <- function(net, on, idx = NULL, layer = NULL, nidx = NULL, nplidx = NULL)
{
    mlp_check_w(net, idx, layer, nidx, nplidx)
    if (is.null(idx)) {
        idx <- mlp_get_w_idx(net, layer, nidx, nplidx)
    }
    output <- .C("mlp_set_active",
                 net@m_layers, net@m_n_pointers,
                 net@m_n_prev, net@m_n_next,
                 net@m_w_pointers, net@m_w_values,
                 net@m_w_flags, net@m_w_on,
                 as.integer(idx), as.integer(on))
    net@m_n_prev <- output[[3]]
    net@m_n_next <- output[[4]]
    net@m_w_values <- output[[6]]
    net@m_w_flags <- output[[7]]
    net@m_w_on <- output[[8]]
    return(net)
}


#' @rdname mlp_net-single-weight-access
#'
#' @export
#'
mlp_set_w <- function(net, val, idx = NULL, layer = NULL, nidx = NULL, nplidx = NULL)
{
    mlp_check_w(net, idx, layer, nidx, nplidx)
    if (is.null(idx)) {
        idx <- mlp_get_w_idx(net, layer, nidx, nplidx)
    }
    if (net@m_w_flags[idx] == 0L) {
        if (!is.null(nidx)) {
            if (nplidx != 0) {
                stop(paste0("connection between neuron ", nidx, " in layer ", layer,
                            " and neuron ", nplidx, " in layer ", layer - 1,
                            " is off"))
            } else {
                stop(paste0("bias of neuron ", nidx, " in layer ", layer,
                            " is off"))
            }
        } else {
            stop(paste0("weight ", idx, " is off"))
        }
    }
    net@m_w_values[idx] <- val
    return(net)
}


#' @rdname mlp_net-single-weight-access
#'
#' @export
#'
mlp_get_w_st <- function(net, idx = NULL, layer = NULL, nidx = NULL, nplidx = NULL)
{
    mlp_check_w(net, idx, layer, nidx, nplidx)
    if (is.null(idx)) {
        idx <- mlp_get_w_idx(net, layer, nidx, nplidx)
    }
    if (net@m_w_flags[idx] != 0L) {
        return(TRUE)
    }
    return(FALSE)
}


#' @rdname mlp_net-single-weight-access
#'
#' @export
#'
mlp_get_w <- function(net, idx = NULL, layer = NULL, nidx = NULL, nplidx = NULL)
{
    mlp_check_w(net, idx, layer, nidx, nplidx)
    if (is.null(idx)) {
        idx <- mlp_get_w_idx(net, layer, nidx, nplidx)
    }
    if (net@m_w_flags[idx] == 0L) {
        if (!is.null(nidx)) {
            if (nplidx != 0) {
                stop(paste0("connection between neuron ", nidx, " in layer ", layer,
                            " and neuron ", nplidx, " in layer ", layer - 1,
                            " is off"))
            } else {
                stop(paste0("bias of neuron ", nidx, " in layer ", layer,
                            " is off"))
            }
        } else {
            stop(paste0("weight ", idx, " is off"))
        }
    }
    return(net@m_w_values[idx])
}


# #########################################################################
# Working with weights' vector
# #########################################################################

#' This function sets network weights to random values drawn from uniform
#' distribution.
#'
#' @param net an object of \code{mlp_net} class
#' @param a numeric value, values will be drawn from uniform distribution
#'        on [-a, a] (by default a = 0.2)
#'
#' @return Network (an object of \code{mlp_net} class) with randomised weights.
#'
#' @export mlp_rnd_weights
#'
mlp_rnd_weights <- function(net, a = 0.2)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    ind <- which(net@m_w_flags != 0L)
    weights <- runif(length(ind), min = -a, max = a)
    net@m_w_values[ind] <- weights
    return(net)
}


#' Set and retrieve (active) weights' values
#'
#' One of FCNN's design objectives (and main advantages) is the complete separation
#' of teaching (and pruning) algorithms from internal network structure workings.
#' This goal is achieved through fast access to (active) weights vector facilitated
#' by FCNN's `compressed' network representation. The following two functions
#' allow users to efficiently retrieve and set network (active) weights vector.
#'
#' @param net an object of \code{mlp_net} class
#' @param weights numeric vector of new active weights' values
#'
#' @return \code{mlp_set_weights} returns network (an object of \code{mlp_net}
#'         class) with active weights set to given values.
#'
#'         \code{mlp_set_weights} returns numeric vector of active weights' values.
#'
#' @name mlp_net-weights-access
#'
#' @export
#'
mlp_set_weights <- function(net, weights)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (length(weights) != net@m_w_on) {
        stop("invalid size of active weights vector")
    }
    ind <- which(net@m_w_flags != 0L)
    net@m_w_values[ind] <- weights
    return(net)
}


#' @rdname mlp_net-weights-access
#'
#' @export
#'
mlp_get_weights <- function(net)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    ind <- which(net@m_w_flags != 0L)
    return(net@m_w_values[ind])
}





# #########################################################################
# Evaluation, MSE, and gradients
# #########################################################################


#' Check validity of inputs and outputs
#'
#' @param net an object of \code{mlp_net} class
#' @param input numeric matrix, each row correspond to one input vector,
#'        number of columns must be equal to the number of neurons
#'        in the network input layer
#' @param output numeric matrix with rows corresponding to expected outputs,
#'        number of columns must be equal to the number of neurons
#'        in the network output layer, number of rows must be equal to the number
#'        of input rows
#' @param i data row index
#'
#' @return This function does not return.
#'
#' @keywords internal
#'
mlp_check_inout <- function(net, input, output = NULL, i = NULL)
{
    if (!is.mlp_net(net)) {
        stop("expected net argument to be of mlp_net class")
    }
    if (!is.numeric(input)) {
        stop("invalid input, expected numeric matrix")
    }
    di <- dim(input)
    if (length(di) != 2) {
        stop("invalid input, expected numeric matrix")
    }
    if (di[1] == 0) {
        stop("input data must have at least one row")
    }
    if (di[2] != net@m_layers[1]) {
        stop("number of input columns does not match the number of input neurons")
    }
    if (!is.null(i)) {
        if ((length(i) != 1) || (!is.integer(i) && !is.numeric(i))) {
            stop("invalid type of row index")
        }
        if ((i < 1) || (i > di[1])) {
            stop("invalid row index")
        }
    }
    if (!is.null(output)) {
        if (!is.numeric(output)) {
            stop("invalid output, expected numeric matrix")
        }
        do <- dim(output)
        if (length(do) != 2) {
            stop("invalid output, expected numeric matrix")
        }
        if (do[1] != di[1]) {
            stop("no. of output rows and no. of input rows disagree")
        }
        if (do[2] != net@m_layers[length(net@m_layers)]) {
            stop("number of output columns does not match the number of output neurons")
        }
    }
}



#' Evaluation
#'
#' Evaluate network output.
#'
#' @param net an object of \code{mlp_net} class
#' @param input numeric matrix, each row corresponds to one input vector,
#'        number of columns must be equal to the number of neurons
#'        in the network input layer
#'
#' @return Numeric matrix with rows representing network outputs corresponding
#'         to input rows.
#'
#' @export mlp_eval
#'
mlp_eval <- function(net, input)
{
    mlp_check_inout(net, input)
    lays <- net@m_layers
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    output <- matrix(0, nrow = nrows, ncol = nout)
    output <- .C("mlp_eval",
                 as.integer(lays), as.integer(length(lays)), as.integer(net@m_n_pointers),
                 as.numeric(net@m_w_values),
                 as.integer(net@m_af_hl), as.numeric(net@m_af_hl_slope),
                 as.integer(net@m_af_ol), as.numeric(net@m_af_ol_slope),
                 as.integer(nrows), as.numeric(input), res = as.numeric(output))$res
    dim(output) <- c(nrows, nout)
    return(output)
}



#' Computing mean squared error, its gradient, and output derivatives
#'
#' The functions use fast FCNN kernel routines and are intended for implementing
#' teaching and pruning algorithms.
#'
#' \code{mlp_mse} returns the mean squared error (MSE). MSE is understood
#' as half of the squared error averaged over all outputs and data records.
#'
#' \code{mlp_grad} computes the gradient of MSE w.r.t. network weights.
#' This function is useful when implementing batch teaching algorithms.
#'
#' \code{mlp_gradi} computes the gradient of MSE w.r.t. network weights at the \code{i}th
#' data record. This is normalised by the number of outputs only,
#' the average over all rows (all i) returns the same as \code{grad(input, output)}.
#' This function is useful when implementing on-line teaching algorithms.
#'
#' \code{mlp_gradij} computes gradients of network outputs,
#' i.e the derivatives of outputs w.r.t. active weights, at given data row.
#' The derivatives of outputs are placed in subsequent columns of the returned
#' matrix. Scaled by the output errors and averaged they give the same
#' as \code{gradi(input, output, i)}. This function is useful in implementing
#' teaching algorithms using second order corrections and Optimal Brain Surgeon
#' pruning algorithm.
#'
#' \code{mlp_jacob} computes the Jacobian of network outputs, i.e the derivatives
#' of outputs w.r.t. inputs, at given data row.
#' The derivatives of outputs are placed in subsequent columns of the returned
#' matrix.
#'
#' @param net an object of \code{mlp_net} class
#' @param input numeric matrix, each row corresponds to one input vector,
#'        number of columns must be equal to the number of neurons
#'        in the network input layer
#' @param output numeric matrix with rows corresponding to expected outputs,
#'        number of columns must be equal to the number of neurons
#'        in the network output layer, number of rows must be equal to the number
#'        of input rows
#' @param i data row index
#'
#' @return \code{mlp_mse} returns mean squared error (numeric value).
#'
#' \code{mlp_grad} returns two-element lists with the first
#' field (\code{grad}) containing numeric vector with gradient and the second
#' (\code{mse}) - the mean squared error.
#'
#' \code{mlp_gradi} returns numeric vector with gradient.
#'
#' \code{mlp_gradij} returns numeric matrix with gradients of outputs in
#' consecutive columns.
#'
#' \code{mlp_jacob} returns numeric matrix with derivatives of outputs in
#' consecutive columns.
#'
#' @name mlp_net-MSE-gradients
#'
#' @export mlp_mse
#'
mlp_mse <- function(net, input, output)
{
    mlp_check_inout(net, input, output = output)
    lays <- net@m_layers
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    mse <- 0
    mse <- .C("mlp_mse",
              as.integer(lays), as.integer(length(lays)), as.integer(net@m_n_pointers),
              as.numeric(net@m_w_values),
              as.integer(net@m_af_hl), as.numeric(net@m_af_hl_slope),
              as.integer(net@m_af_ol), as.numeric(net@m_af_ol_slope),
              as.integer(nrows), as.numeric(input),
              as.numeric(output), res = mse)$res
    return(mse)
}



#' @rdname mlp_net-MSE-gradients
#'
#' @export
#'
mlp_grad <- function(net, input, output)
{
    mlp_check_inout(net, input, output = output)
    lays <- net@m_layers
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    grad <- numeric(length = net@m_w_on + 1)
    grad <- .C("mlp_grad",
               lays, length(lays), net@m_n_pointers,
               net@m_w_pointers, net@m_w_flags, net@m_w_values,
               net@m_af_hl, net@m_af_hl_slope,
               net@m_af_ol, net@m_af_ol_slope,
               as.integer(nrows), as.numeric(input), as.numeric(output),
               res = grad)$res
    mse <- grad[1]
    grad <- grad[2:length(grad)]
    return(list(grad = grad, mse = mse))
}



#' @rdname mlp_net-MSE-gradients
#'
#' @export
#'
mlp_gradi <- function(net, input, output, i)
{
    mlp_check_inout(net, input, output = output, i = i)
    lays <- net@m_layers
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    grad <- numeric(length = net@m_w_on)
    grad <- .C("mlp_gradi",
               lays, length(lays), net@m_n_pointers,
               net@m_w_pointers, net@m_w_flags, net@m_w_values,
               net@m_af_hl, net@m_af_hl_slope,
               net@m_af_ol, net@m_af_ol_slope,
               as.integer(nrows), as.integer(i), as.numeric(input), as.numeric(output),
               res = grad)$res
    return(grad)
}



#' @rdname mlp_net-MSE-gradients
#'
#' @export
#'
mlp_gradij <- function(net, input, i)
{
    mlp_check_inout(net, input, i = i)
    lays <- net@m_layers
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    grad <- matrix(0, nrow = net@m_w_on, ncol = nout)
    grad <- .C("mlp_gradij",
               lays, length(lays), net@m_n_pointers,
               net@m_w_pointers, net@m_w_flags,
               net@m_w_values, net@m_w_on,
               net@m_af_hl, net@m_af_hl_slope,
               net@m_af_ol, net@m_af_ol_slope,
               as.integer(nrows), as.integer(i), as.numeric(input),
               res = grad)$res
    dim(grad) <- c(net@m_w_on, nout)
    return(grad)
}



#' @rdname mlp_net-MSE-gradients
#'
#' @export
#'
mlp_jacob <- function(net, input, i)
{
    mlp_check_inout(net, input, i = i)
    lays <- net@m_layers
    nin <- lays[1]
    nout <- lays[length(lays)]
    nrows <- dim(input)[1]
    jacob <- matrix(0, nrow = nin, ncol = nout)
    jacob <- .C("mlp_jacob",
                lays, length(lays), net@m_n_pointers,
                net@m_w_pointers, net@m_w_flags,
                net@m_w_values, net@m_w_on,
                net@m_af_hl, net@m_af_hl_slope,
                net@m_af_ol, net@m_af_ol_slope,
                as.integer(nrows), as.integer(i), as.numeric(input),
                res = jacob)$res
    dim(jacob) <- c(nin, nout)
    return(jacob)
}






