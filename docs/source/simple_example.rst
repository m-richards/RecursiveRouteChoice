A Simple Example
================

Do the network from the fosgerau paper?


.. note::
   The example presented demonstrates the reverse order to a typical use case. Here we
simulate observations,
   picking arbitrary parameters and then estimate parameters from those observations. This is nice
   for a simple example since then we don't need an external data source.

Prediction
----------

First we illustrate how to simulate observations on a trivial network, with the arbitrary
parameter weight for the single network attribute; distance being :math:`\beta=-0.4`.

::

    import numpy as np
    from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct

    # DATA
    # A trivial network
    distances = np.array(
        [[0, 5, 0, 4],
         [0, 0, 6, 0],
         [0, 6, 0, 5],
         [4, 0, 0, 0]])

    incidence_mat = (distances > 0).astype(int)

    network_attribute_list = [distances]
    network_struct = ModelDataStruct(network_attribute_list, incidence_mat,
                                     data_array_names_debug=("distances",))
    model = RecursiveLogitModelPrediction(network_struct, initial_beta=[-0.4], mu=1)

    obs_indices = [0, 3]
    dest_indices = [1, 2]
    obs_per_pair = 15
    print(f"Generating {obs_per_pair * len(obs_indices) * len(dest_indices)} obs total")

    obs = model.generate_observations(origin_indices=obs_indices, dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=1)
    print(obs)

The code is hopefully rather self explanatory. We supply the input data and incidence matrix to the
:code:`ModelDataStruct` which provides a generic interface for the model to access network
attributes through. To predict, we initialise the model, supplying the value for :math:`\beta`.
Finally, we simulate trips between the provided arcs on the network, with repetition.

Estimation
----------
Now we follow on from the above example, reusing the same network, and assume that we are trying to
estimate the parameter :math:`\beta` from :code:`obs` as generated above.

::

    import optimisers
    from recursive_route_choice import RecursiveLogitModelEstimation

    optimiser = optimisers.ScipyOptimiser(method='bfgs')

    model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=[-5], mu=1,
                                          optimiser=optimiser)
    beta = model.solve_for_optimal_beta(verbose=False)
    print("Optimal beta determined was", beta)

The estimation code is also very simple. We declare the optimiser class to use, supply it to the
model, with some initial guess for :math:`\beta` and then solve, which hopefully will converge.
