import numpy as np
import awkward1 as ak
from scipy.sparse import dok_matrix

from recursiveRouteChoice.data_loading import load_obs_from_json

from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500
obs_fname = 'my_networks_obs.json'
obs_lil = load_obs_from_json(obs_fname)
obs_ak = ak.from_json(obs_fname)
print("len ", len(obs_ak))


# silly levels of inefficiency but will fix later

# obs = np.array(obs_lil)
# obs = scipy.sparse.dok_matrix(obs_lil)

#
# DATA
distances = np.array(
    [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
     [3.5, 3, 4, 0, 2.5, 3, 3, 0],
     [4.5, 4, 5, 0, 0, 0, 4, 3.5],
     [3, 0, 0, 2, 2, 2.5, 0, 2],
     [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
     [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
     [0, 3, 4, 0, 2.5, 3, 3, 2.5],
     [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

distances = dok_matrix(distances)

incidence_mat = (distances > 0).astype(int)

data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))

beta_vec = np.array([-114])  # 4.96 diverges
import time

optimiser = optimisers.LineSearchOptimiser(optimisers.OptimHessianType.BFGS, max_iter=40)
# optimiser = op.ScipyOptimiser(method='bfgs')

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
# log_like_out, grad_out = model.get_log_likelihood()
a = time.time()
model.solve_for_optimal_beta()
print(time.time() - a)