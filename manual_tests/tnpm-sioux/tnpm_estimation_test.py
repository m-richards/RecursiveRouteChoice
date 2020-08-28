import scipy
import numpy as np
import awkward1 as ak
from data_loading import write_obs_to_json, load_obs_from_json
import optimisers as op
from data_loading import load_tnpm_to_sparse

from main import RecursiveLogitDataStruct, RecursiveLogitModelPrediction, \
    RecursiveLogitModelEstimation

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500
obs_fname = 'sioux_out_obs.json'
obs_lil = load_obs_from_json(obs_fname)
obs_ak = ak.from_json(obs_fname)
print("len ", len(obs_ak))


# silly levels of inefficiency but will fix later

# obs = np.array(obs_lil)
# obs = scipy.sparse.dok_matrix(obs_lil)

#
# DATA
network_file = "SiouxFalls_net.tntp"
arc_to_index_map, distances = load_tnpm_to_sparse(network_file, columns_to_extract=["length"])

index_node_pair_map = {v:k for (k,v) in arc_to_index_map.items()}

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = RecursiveLogitDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))

beta_vec = np.array([-1])
optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
# log_like_out, grad_out = model.get_log_likelihood()

model.solve_for_optimal_beta()


