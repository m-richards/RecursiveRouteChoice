
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from data_loading import write_obs_to_json
from data_loading import load_tnpm_to_sparse

from main import RecursiveLogitDataStruct, RecursiveLogitModelPrediction, \
    RecursiveLogitModelEstimation

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500


# DATA

# distances = np.array(
#     [[4, 3.5, 3],
#      [3.5, 3, 2.5],
#      [3, 2.5, 2],
#      ])
distances = np.array(
    [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
     [3.5, 3, 4, 0, 2.5, 3, 3, 0],
     [4.5, 4, 5, 0, 0, 0, 4, 3.5],
     [3, 0, 0, 2, 2, 2.5, 0, 2],
     [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
     [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
     [0, 3, 4, 0, 2.5, 3, 3, 2.5],
     [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

incidence_mat = (distances > 0).astype(int)


data_list = [distances]
network_struct = RecursiveLogitDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances"))

beta_vec = np.array([-16])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
# obs_indices = [i for i in range(8)]
# obs = model.generate_observations(origin_indices=obs_indices,
#                                   dest_indices=obs_indices,
#                                   num_obs_per_pair=1, iter_cap=2000, rng_seed=1,
#                                   )
obs_indices =[1,2,3,4,5,6,7,8]
obs = model.generate_observations(origin_indices=obs_indices,
                                  dest_indices=[7,3],
                                  num_obs_per_pair=20, iter_cap=2000, rng_seed=1,
                                  )

print(obs)


print("\nPath in terms of arcs:")
for path in obs:
    string = "Orig: "
    f = "Empty Path, should not happen"
    for arc_index in path:
        string += f"-{arc_index + 1}- => "
    string += f": Dest"

    print(string)

obs_fname = "my_networks_obs2.json"
write_obs_to_json(obs_fname, obs, allow_rewrite=True)


import scipy
import numpy as np
import awkward1 as ak
from scipy.sparse import dok_matrix

from data_loading import write_obs_to_json, load_obs_from_json
import optimisers as op
from data_loading import load_tnpm_to_sparse

from main import RecursiveLogitDataStruct, RecursiveLogitModelPrediction, \
    RecursiveLogitModelEstimation

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
network_struct = RecursiveLogitDataStruct(data_list, incidence_mat,
                                          data_array_names_debug=("distances",))

beta = -5
beta_vec = np.array([beta]) # 4.96 diverges
optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=40)

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
log_like_out, grad_out = model.get_log_likelihood()
print("LL1:", log_like_out, grad_out)
h = 0.0002
model.update_beta_vec([beta+h])
ll2, grad2 = model.get_log_likelihood()
print("LL2:", ll2, grad2)
print("finite difference:", (ll2- log_like_out)/h)


ls_out = model.solve_for_optimal_beta()
print(model.optim_function_state.val_grad_function(beta))
#=======================================================
print(120 * "=", 'redo with scipy')
optimiser = op.ScipyOptimiser(method='bfgs')

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs_ak,
                                      initial_beta=beta_vec, mu=1,
                                      optimiser=optimiser)
log_like_out, grad_out = model.get_log_likelihood()
print("LL1:", log_like_out, grad_out)
h = 0.0002
model.update_beta_vec([beta+h])
ll2, grad2 = model.get_log_likelihood()
print("LL2:", ll2, grad2)
print("finite difference:", (ll2- log_like_out)/h)


ls_out = model.solve_for_optimal_beta(verbose=True)
print(model.optim_function_state.val_grad_function(beta))




print(model.optim_function_state.val_grad_function(beta))

from scipy import optimize as scopt
# out =scopt.minimize_scalar(lambda x: model.optim_function_state.function(x)[0],
#                       )
out_bfgs = scopt.minimize(lambda x: model.optim_function_state.val_grad_function(x)[0],
                          x0=np.array([-5]), method='BFGS',
                          jac=lambda x: model.optim_function_state.val_grad_function(x)[1],
                          )
# print("scipy out")
# print(out_bfgs)

def fun_wrapper(x):
    return model.optim_function_state.val_grad_function(x)[0]
def grad_wrapper(x):
    return model.optim_function_state.val_grad_function(x)[1]

finite_diff_options = {}
def hess_wrapper(x):
    return approx_derivative(grad_wrapper, x, '2-point', **finite_diff_options)


out_trust_ncg = scopt.minimize(fun_wrapper,
                          x0=np.array([-5]), method='trust-ncg', hess=hess_wrapper,
                          jac=grad_wrapper,

                               )

print("ls x", ls_out, "bfgs_out", out_bfgs.x, "trst_ncg", out_trust_ncg.x)

#
#
# model.update_beta_vec(np.array([-16]))
# print(model.get_log_likelihood())


