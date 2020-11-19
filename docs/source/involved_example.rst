Sioux Falls Example
===================
Here we present example usage for the classic Sioux Falls network, again estimating parameters
from simulated data. This is the same example appearing in the README, but we give a little more
detail.

.. literalinclude:: /../tests/docs/test_sioux_falls_example.py
    :lines: 8-20

Here we see usage of one of the convenience loader methods provided for the tntp format. Since
the network is very small and easily fits in memory, we store it in a dense representation.

.. literalinclude:: /../tests/docs/test_sioux_falls_example.py
    :lines: 23,25-26

This is the standardised way of passing data to the model, the network attributes are collected
into a :code:`ModelDataStruct` instance, alongside the corresponding incidence matrix. We then
simulate a series of observations between every origin destination pair, note that whilst this
will print there are 576 observations, there will actually be less as observations starting and
ending at the same node are omitted.

.. literalinclude:: /../tests/docs/test_sioux_falls_example.py
    :lines: 28-41

We now construct the estimation model to attempt to recover the parameters the data was simulated
with. We construct and optimiser instance, this time using L-BFGS and provide an initial iterate
for the optimisation algorithm. Note that these are of differing orders of magnitude, due to
capacity being measured on the scale of tens of thousands for Sioux Falls. Attempting to use
:code:`beta_est_init = [-5, -5]` will fail as the matrix :math:`M` will be numerically zero,
giving rise to a degenerate solution. This case is explicitly caught as an error by the code.


.. literalinclude:: /../tests/docs/test_sioux_falls_example.py
    :lines: 46-55

Running the code, we get that on the specified seed, so the original parameters have been
recovered somewhat well. If we increase to simulating a trip between every origin and
destination, we see more accurate recovery of the parameters.

::

    # Every second OD pair:
    beta expected: [-0.8000, -0.0001], beta_actual: [-1.021, -0.000175]

    # One simulation per OD pair:
    beta expected: [-0.8000, -0.0001], beta_actual: [-0.7963, -0.0002]
