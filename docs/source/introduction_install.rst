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
using the main estimation script, :code:`RLoptimizer` immediately crashes due to hard coded minimum
dimensions,
and the simulation code :code:`createSimulatedObs` crashes due some some change in
version where "Value" has become "value").

This package intends to address some of the shortcomings of this implementation, to list a few key
points

* Simple examples work out of the box, with a clearly specified data format
* Code is documented consistently, with explanatory introductory material
* Code is designed in a modular, clear manner to aid in understanding and extensibility
* Some mathematical correction to the algorithm implementation

The package has been developed as part of my (maths) honours thesis and subsequently, significant
emphasis
has been placed on ensuring the mathematical correctness of the components.

Whilst the shortcomings of Tien Mai's code have been noted, it has still been an important reference
in the development of this package. In fact, the original checks of consistency were made with
respect to a corrected version of these scripts. At this point however, RecursiveRouteChoice should
be
considered completely independent. The only exception is the inclusion of a direct port of the line
search code (but this is slow since python loops are slow), but this is deprecated in favour of
the interface to SciPy solvers.

It also should be acknowledged that the results of our numerical experiments are notably less
optimistic about the applicability of the recursive logit than prior literature. From our
investigation there are still fundamental nontrivial issues, which would need to be resolved to
allow any kind of reasonable adoption of these models. We hope that this disparity can be explained
by either the arc based formulation with prohibition of u-turns employed in prior research, or some
of the mathematical technicalities which have been altered in this implementation. If this is not
the case however it would suggest there is an inconsistency between been our results and those which
have been published prior.

Installation
============
See the Github readme for now.


