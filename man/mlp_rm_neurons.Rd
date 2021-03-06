% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mlp_net.R
\name{mlp_rm_neurons}
\alias{mlp_rm_neurons}
\title{Remove redundant neurons in a multilayer perceptron network}
\usage{
mlp_rm_neurons(net, report = FALSE)
}
\arguments{
\item{net}{an object of \code{mlp_net} class}

\item{report}{logical value, if TRUE, information about removed neurons
will be printed on the console (FALSE by default)}
}
\value{
Three-element list. The first element (\code{net}) is the network
        (an object of \code{mlp_net} class) with all redundant neurons
        removed, the second (\code{ncount}) - the number of neurons removed,
        the third (\code{wcount}) - the number of weights removed.
}
\description{
This function removes redundant neurons from the network, i.e. hidden layers'
neurons that are not connected to neurons in the previous layer or the next
layer. If a neuron is not connected to neurons in the previous layer but
is connected to neurons in the next layer (effectively acts as an additional
bias), biases of neurons in the next layer are properly adjusted, therefore,
the resulting network behaves just like the initial one.
}

