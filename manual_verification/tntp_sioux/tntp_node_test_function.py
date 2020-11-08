import numpy as np
from scipy.sparse import dok_matrix
import awkward1 as ak

from data_loading import write_obs_to_json, load_obs_from_json, load_tntp_to_sparse_arc_formulation, \
    load_tntp_to_sparse_node_formulation
from recursive_logit_efficient_update import RecursiveLogitModelEstimationSM
from recursive_route_choice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation

import optimisers as op

np.set_printoptions(edgeitems=10, linewidth=300)

import textwrap
# np.core.arrayprint._line_width = 500

# DATA
wrapper = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t",width=120)

def consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0,
                     test_range=None):
    if test_range is None:
        test_range = np.arange(-0.1, -2.1, -0.1)
    # network_file = "EMA_net.tntp"

    data_list, data_list_names = load_tntp_to_sparse_node_formulation(network_file,
                                                                      columns_to_extract=["length",
                                                                                          ],
                                                                      )
    distances = data_list[0]

    incidence_mat = (distances > 0).astype(int)

    network_struct = ModelDataStruct(data_list, incidence_mat,
                                     data_array_names_debug=("distances", "u_turn"))

    beta_vec = np.array([-0.1])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec, mu=1)
    print("Linear system size", model.get_exponential_utility_matrix().shape)

    print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
          f"beta sim val")

    def get_data(beta_vec, seed=None):
        beta_vec_generate = np.array([beta_vec])
        model = RecursiveLogitModelPrediction(network_struct,
                                              initial_beta=beta_vec_generate, mu=1)
        obs = model.generate_observations(origin_indices=orig_indices,
                                          dest_indices=dest_indices,
                                          num_obs_per_pair=obs_per_pair, iter_cap=2000,
                                          rng_seed=seed,
                                          )
        return obs

    optimiser = op.ScipyOptimiser(method='l-bfgs-b')  # bfgs, l-bfgs-b

    import time
    a = time.time()
    expected = []
    actual = []
    for n, beta_gen in enumerate(test_range, start=1):
        expected.append(beta_gen)
        try:
            obs = get_data(beta_gen, seed=None)
        except ValueError as e:
            print(f"beta = {beta_gen} failed, {e}")
            actual.append(0.0)
            continue
        # print(obs)
        beta0 = -5
        model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=beta0, mu=1,
                                              optimiser=optimiser)
        beta = model.solve_for_optimal_beta(verbose=False)
        actual.append(float(beta))
        print("beta_expected", beta_gen, "beta actual", beta, "\nOBS:")
        # text_list = wrapper.wrap(str(obs))
        # print("\n".join(text_list))

    b = time.time()
    print("elapsed =", b - a, "s")
    return np.array(expected), np.array(actual)

if __name__ =="__main__":
    beta0 = -5
    network_file = "SiouxFalls_net.tntp"
    orig_indices = np.arange(0, 23, 6)
    dest_indices = np.arange(1, 23, 6)
    obs_per_pair = 1
    consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0)