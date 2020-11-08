Sioux Falls Example
===================
Here we present example usage for the classic Sioux Falls network, again estimating parameters
from simulated data. This is the same example appearing in the README, but we give a little more
detail.

::

    import numpy as np
    from data_loading import load_tntp_node_formulation
    from recursive_route_choice import RecursiveLogitModelPrediction, ModelDataStruct, \
        RecursiveLogitModelEstimation
    import optimisers as op

    # DATA
    network_file = "SiouxFalls_net.tntp"
    node_max = 24  # from network file

    data_list, data_list_names = load_tntp_node_formulationn(
        network_file, columns_to_extract=["length", "capacity"], sparse_format=False)

Here we see usage of one of the convenience loader methods provided for the tntp format. Since
the network is very small and easily fits in memory, we store it in a dense representation.

::

    distances = data_list[0]
    incidence_mat = (distances > 0).astype(int)
    network_struct = ModelDataStruct(data_list, incidence_mat)

This is the standardised way of passing data to the model, the network attributes are collected
into a :code:`ModelDataStruct` instance, alongside the corresponding incidence matrix. We then
simulate a series of observations between every origin destination pair, note that whilst this
will print there are 576 observations, there will actually be less as observations starting and
ending at the same node are omitted.

::

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

We now construct the estimation model to attempt to recover the parameters the data was simulated
with. We construct and optimiser instance, this time using L-BFGS and provide an initial iterate
for the optimisation algorithm. Note that these are of differing orders of magnitude, due to
capacity being measured on the scale of tens of thousands for Sioux Falls. Attempting to use
:code:`beta_est_init = [-5, -5]` will fail as the matrix :math:`M` will be numerically zero,
giving rise to a degenerate solution. This case is explicitly caught as an error by the code.

::

    optimiser = op.ScipyOptimiser(method='l-bfgs-b')
    beta_est_init = [-5, -0.00001]
    model_est = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta_est_init, mu=1,
                                              optimiser=optimiser)

    beta = model_est.solve_for_optimal_beta(verbose=False)

    print(f"beta expected: [{beta_sim[0]:6.4f}, {beta_sim[1]:6.4f}],"
                  f" beta_actual: [{beta[0]:6.4f}, {beta[1]:6.4f}]")


Running the code, we get that on the specified seed, so the original parameters have been
recovered reasonably well. Note that changing the number of observations available to estimate from
will
have a non-trivial impact on the output results if there are not enough.

::

    beta expected: [-0.8000, -0.0001], beta_actual: [-0.7963, -0.0002]
