import numpy as np
from scipy.sparse import dok_matrix
import awkward1 as ak

from recursiveRouteChoice.data_loading import write_obs_to_json, load_obs_from_json
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers


np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
# silly deterministic network
# distances = np.array(
#     [[0, 5, 0, 4],
#      [0, 0, 6, 0],
#      [0, 6, 0, 5],
#      [4, 0, 0, 0]])
# extended silly network
distances = np.array(
    [[0, 5, 0, 4, 0, 0, 0, 0, 0, 0],
     [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
     [0, 6, 0, 5, 0, 0, 0, 0, 0, 0],
     [4, 0, 0, 0, 5, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
     [5, 0, 0, 0, 6, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 6, 6, 0],
     [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
     [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
     [0, 0, 0, 0, 0, 0, 0, 6, 6, 0]
     ])


# distances = np.array(
#     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
#      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
#      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
#      [3, 0, 0, 2, 2, 2.5, 0, 2],
#      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
#      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
#      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
#      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances"))
beta_known = -0.4
beta_vec_generate = np.array([beta_known])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec_generate, mu=1)

obs_indices = [0]
obs = model.generate_observations(origin_indices=obs_indices,
                                  dest_indices=[1],
                                  num_obs_per_pair=20, iter_cap=2000, rng_seed=1,
                                  )

print(obs)

print("\nPath in terms of arcs:")
for path in obs:
    string = "Orig: "
    f = "Empty Path, should not happen"
    for arc_index in path[1:]:
        string += f"-{arc_index + 1}- => "
    string += ": Dest"

    print(string)

obs_fname = "my_networks_obs2.json"
write_obs_to_json(obs_fname, obs, allow_rewrite=True)

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500
# obs_fname = 'my_networks_obs.json'
obs_lil = load_obs_from_json(obs_fname)
obs_ak = ak.from_json(obs_fname)
print("len ", len(obs_ak))


# silly levels of inefficiency but will fix later

# obs = np.array(obs_lil)
# obs = scipy.sparse.dok_matrix(obs_lil)

#
# DATA
# distances = np.array(
#     [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
#      [3.5, 3, 4, 0, 2.5, 3, 3, 0],
#      [4.5, 4, 5, 0, 0, 0, 4, 3.5],
#      [3, 0, 0, 2, 2, 2.5, 0, 2],
#      [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
#      [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
#      [0, 3, 4, 0, 2.5, 3, 3, 2.5],
#      [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

distances = dok_matrix(distances)

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = ModelDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))


# optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=40)
#
# model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
#                                       initial_beta=beta_vec, mu=1,
#                                       optimiser=optimiser)
# log_like_out, grad_out = model.get_log_likelihood()
# print("LL1:", log_like_out, grad_out)
# h = 0.0002
# model.update_beta_vec([beta+h])
# ll2, grad2 = model.get_log_likelihood()
# print("LL2:", ll2, grad2)
# print("finite difference:", (ll2- log_like_out)/h)
#
#
# ls_out = model.solve_for_optimal_beta()
# print(model.optim_function_state.val_grad_function(beta))
# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = optimisers.ScipyOptimiser(method='bfgs')
beta = -0.4
beta_vec = np.array([beta])  # 4.96 diverges
model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
log_like_out, grad_out = model.get_log_likelihood()
print("start beta", beta, "log likelihood:", log_like_out)

# beta_out = model.solve_for_optimal_beta(verbose=True)
# print("optimal solve complete")
# # print(model.optim_function_state.val_grad_function(beta))
# print("start beta", beta, "likelihood:", np.exp(-log_like_out))
# print("best beta", beta_out, "likelihood:", np.exp(-model.optim_function_state.value))
# print("best beta known", beta_known, "likelihood:", np.exp(-model.eval_log_like_at_new_beta(
#     beta_known)[0]))
# print("best beta incorrect", model.get_beta_vec())
#
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
beta = np.arange(-0.1, -1, -0.05)
like = np.zeros(len(beta))
log_like = np.zeros(len(beta))

for i in range(len(beta)):
    like[i] = np.exp(-model.eval_log_like_at_new_beta(beta[i])[0])
    log_like[i] = -model.eval_log_like_at_new_beta(beta[i])[0]
plt.scatter(beta, log_like, label=rf'$\beta_{{true}} = {beta_known}$')
plt.legend()

print("actually generated with beta= ", beta_known, "like = ", np.exp(
    -model.eval_log_like_at_new_beta(beta_known)[0]))

plt.show()
# model.update_beta_vec(np.array([-16]))
# print(model.get_log_likelihood())
