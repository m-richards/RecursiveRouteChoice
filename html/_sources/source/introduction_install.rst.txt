Overview
========
This package implements the recursive logit model first developed in
`Fosgerau, Frejinger & Karlstrom 2013 <https://doi.org/10.1016/j.trb.2013.07.012>`_
with further developments in papers Tien Mai. There are two key components:

* | **Estimation -** Given a series of observations on a network and a set of network attributes,
  | estimate the parameter coefficients of each attribute

* | **Simulation/ Prediction -** Given a set of network attributes and parameter estimates for each
  | attribute, simulate observations on the network consistent to the parameter values

`Tien Mai has previously released an implementation <https://github.com/maitien86>`_
for the recursive logit and
more advanced variants which are not (currently) implemented here. Unfortunately, this code is
in various states of ill-repair (for instance running the :code:`RecursiveLogit.Classical.V2` repo
using the estimation script, :code:`RLoptimizer` immediately crashes due to hard coded minimum
dimensions,
and the simulation code :code:`createSimulatedObs` crashes due some some change in
version where "Value" has become "value").

This package intends to address some of the shortcomings of this implementation, to list a few key
points
* Simple examples work out of the box, with a clearly specified data format
* Code is documented consistently, with explanatory introductory material
* Code is designed in a modular, clear main to aid in understanding and extensibility

The package has been developed as part of my honours thesis and subsequently, significant emphasis
has been placed on ensuring the mathematical correctness of the components.

Whilst the shortcomings of Tien Mai's code have been noted, it has still been an important reference
in the development of this package. In fact, the original checks of consistency were made with
respect to a corrected version of these scripts. At this point however, RecursiveRouteChoice should
be
considered completely independent. The only exception is the inclusion of a direct port of the line
search code (but this is slow since python loops are slow), but this is deprecated in favour of
the interface to SciPy solvers.


Installation
============
Currently one can install from the repository directly using pip::

   pip install git+https://github.com/m-richards/RecursiveLogit.git

Hopefully this will also appear on PyPi at some point.


