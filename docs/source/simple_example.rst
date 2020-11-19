A Simple Example
================

Do the network from the fosgerau paper?


.. note::
    The example presented demonstrates the reverse order to a typical use case. Here we
    simulate observations, picking arbitrary parameters and then estimate parameters from those
    observations. This is nice for a simple example since then we don't need an external data source.

Prediction
----------

First we illustrate how to simulate observations on a trivial network, with the arbitrary
parameter weight for the single network attribute; distance being :math:`\beta=-0.4`.

.. literalinclude:: /../tests/docs/test_simple_example.py
    :lines: 5-30

The code is hopefully rather self explanatory. We supply the input data and incidence matrix to the
:code:`ModelDataStruct` which provides a generic interface for the model to access network
attributes through. To predict, we initialise the model, supplying the value for :math:`\beta`.
Finally, we simulate trips between the provided arcs on the network, with repetition.

Estimation
----------
Now we follow on from the above example, reusing the same network, and assume that we are trying to
estimate the parameter :math:`\beta` from :code:`obs` as generated above.


.. literalinclude:: /../tests/docs/test_simple_example.py
    :lines: 33-42

The estimation code is also very simple. We declare the optimiser class to use, supply it to the
model, with some initial guess for :math:`\beta` and then solve, which hopefully will converge.
