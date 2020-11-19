RecursiveRouteChoice
====================
RecursiveRouteChoice is an efficient  Python implementation of the recursive logit model for
route choice. This kind of model was first developed in
`Fosgerau, Frejinger & Karlstrom 2013 <https://doi.org/10.1016/j.trb.2013.07.012>`_

This implementation contains both prediction and estimation models. `See the docs
for a more detailed introduction <https://m-richards.github.io/RecursiveRouteChoice>`_. Here is a
quick start overview.

Installation
------------
Currently one can install from the repository directly using pip::

   pip install git+https://github.com/m-richards/RecursiveLogit.git
   pip install -r requirements.txt

It is recommended to use some kind of virtual environment to avoid conflicting package versions.
There is also a :code:`requirements_strict.txt` which explicitly specifies package versions. This
will likely contain less up to date versions, but should always work regardless of future
api changes of the dependencies.

Example Usage
-------------

::

    import numpy as np
    from load_tntp_node_formulation
    from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers

    # DATA
    network_file = "SiouxFalls_net.tntp"
    node_max = 24  # from network file

    data_list, data_list_names = load_tntp_node_formulation(
        network_file, columns_to_extract=["length", "capacity"], sparse_format=False)

    distances = data_list[0]

    incidence_mat = (distances > 0).astype(int)
    network_struct = ModelDataStruct(data_list, incidence_mat)

    beta_sim = np.array([-0.8, -0.00015])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_sim, mu=1)
    print("Linear system size", model.get_exponential_utility_matrix().shape)

    orig_indices = np.arange(0, node_max, 1)
    # dest_indices = (orig_indices + 5) % node_max
    dest_indices = np.arange(0, node_max, 1)  # sample every OD pair
    obs_per_pair = 1
    print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
          f"configuration")
    seed = 42
    obs = model.generate_observations(origin_indices=orig_indices, dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed)

    optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')
    beta_est_init = [-5, -0.00001]
    model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_est_init, mu=1,
                                              optimiser=optimiser)

    beta = model_est.solve_for_optimal_beta(verbose=False)

    print(f"beta expected: [{beta_sim[0]:6.4f}, {beta_sim[1]:6.4f}],"
                  f" beta_actual: [{beta[0]:6.4f}, {beta[1]:6.4f}]")
