import numpy as np

from recursiveRouteChoice.data_loading import load_tntp_node_formulation
from recursiveRouteChoice import RecursiveLogitModelPrediction, ModelDataStruct, \
    RecursiveLogitModelEstimation, optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
network_file = "SiouxFalls_net.tntp"
# network_file = "EMA_net.tntp"

data_list, data_list_names = load_tntp_node_formulation(network_file,
                                                        columns_to_extract=["length",
                                                                                     ],
                                                        )
# print(arc_to_index_map)
# print(data_list, data_list_names)
distances = data_list[0]

incidence_mat = (distances > 0).astype(int)

network_struct = ModelDataStruct(data_list, incidence_mat,
                                 data_array_names_debug=("distances", "u_turn"))

beta_vec = np.array([-0.1])
model = RecursiveLogitModelPrediction(network_struct,
                                      initial_beta=beta_vec, mu=1)
print("Linear system size", model.get_exponential_utility_matrix().shape)
# orig_indices = np.arange(0, 74, 16)
# dest_indices = np.arange(0, 74, 8)
orig_indices = np.arange(0, 23, 6)
dest_indices = np.arange(1, 23, 6)
obs_per_pair = 1

# orig_indices = np.arange(0, 23, 12)
# dest_indices = np.arange(1, 23, 12)
# obs_per_pair = 1
#

print(f"Generating {obs_per_pair * len(orig_indices) * len(dest_indices)} obs total per "
      f"configuration")


def get_data(beta_vec, seed=None):
    beta_vec_generate = np.array([beta_vec])
    model = RecursiveLogitModelPrediction(network_struct,
                                          initial_beta=beta_vec_generate, mu=1)
    obs = model.generate_observations(origin_indices=orig_indices,
                                      dest_indices=dest_indices,
                                      num_obs_per_pair=obs_per_pair, iter_cap=2000, rng_seed=seed,
                                      )
    return obs


# =======================================================
print(120 * "=", 'redo with scipy')
optimiser = optimisers.ScipyOptimiser(method='l-bfgs-b')  # bfgs, l-bfgs-b



import time
a = time.time()
expected = []
actual = []
for n, beta_gen in enumerate(np.arange(-0.1, -2.1, -0.1), start=1):
    expected.append(beta_gen)
    try:
        obs = get_data(beta_gen, seed=2)
    except ValueError as e:
        print(f"beta = {beta_gen} failed, {e}")
        actual.append(0.0)
        continue
    # print(obs)
    beta_init = -5
    model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                          initial_beta=beta_init, mu=1,
                                          optimiser=optimiser)
    beta = model.solve_for_optimal_beta(verbose=False)
    actual.append(float(beta))
    print("beta_expected", beta_gen, "beta actual", beta)

b = time.time()
print("elapsed =", b-a, "s")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
actual = np.array(actual)
expected = np.array(expected)
print(list(expected))
print(list(actual))

from brokenaxes import brokenaxes


output = np.abs((expected - actual)/expected)
index_failures = (actual == 0)
print("expected", expected)
print("actual", actual)
print("rel error", output)
# plt.scatter(expected[index_failures], actual[index_failures], marker='x', label="Simulation Failed")
# plt.scatter(expected[~index_failures], actual[~index_failures], label="Valid Results")


ylim = [.05, -2.25]
ylim2 = [-12.25, -14]
bax = brokenaxes(ylims=(ylim, ylim2), hspace=.05, fig=plt.figure(figsize=(8,6)))
bax.scatter(expected[index_failures], actual[index_failures], marker='x', label="Simulation Failed")
bax.scatter(expected[~index_failures], actual[~index_failures], label="Valid Results")
x = np.linspace(0, np.min(actual), 10)
bax.plot(x, x, color='k')
bax.set_xlim([0, -2.1])
bax.set_xlabel(r"Value of $\beta$ used to Simulate")
bax.set_ylabel(r"Value of $\beta$ obtained from Estimation")
bax.legend()
plt.show()

# import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes
# import numpy as np

# fig = plt.figure(figsize=(5,2))
# bax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05)
# x = np.linspace(0, 1, 100)
# bax.plot(x, np.sin(10 * x), label='sin')
# bax.plot(x, np.cos(10 * x), label='cos')
# bax.legend(loc=3)
# bax.set_xlabel('time')
# bax.set_ylabel('value')




# Kind of works with respect to scaling but has a label i can't get rid of
# fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# ylim  = [0, -2]
# ylim2 = [-12.8, -14]
# import matplotlib.gridspec as gridspec
#
#
# ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
# ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
# gs = gridspec.GridSpec(2, 1, height_ratios=[ylim2ratio, ylimratio])
# fig = plt.figure()
# ax = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1], sharex=ax)
#
# plt.subplots_adjust(hspace=0.034)
#
# # f.tight_layout()
# ax.scatter(expected, actual)
# x = np.linspace(0,np.min(actual),10)
# ax.plot(x,x, color='k')
# ax2.plot(x,x, color='k')
# ax2.scatter(expected, actual)
# ax2.set_ylim(0, -2)
# ax.set_ylim(*ylim2)
# ax.set_yticks([-13, -13.5, -14])
#
#
# # hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # ax.xaxis.tick_top()
# # ax.xaxis.label.set_color('white')
#
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# # ax.get_xaxis().set_visible(False)
# ax2.xaxis.tick_bottom()
# ax.set_xlim([expected.max()+0.1, expected.min()-0.2])
#
#
#
# # plt.xlim([expected.max()+0.1, expected.min()-0.2])
# # plt.ylim([actual.max()+0.1, actual.min()-0.2])
# ax2.set_xlabel(r"Value of $\beta$ used to Simulate")
# ax2.set_ylabel("\t\t\t\t" + r"Value of $\beta$ obtained from Estimation")
# # This looks pretty good, and was fairly painless, but you can get that
# # cut-out diagonal lines look with just a bit more work. The important
# # thing to know here is that in axes coordinates, which are always
# # between 0-1, spine endpoints are at these locations (0,0), (0,1),
# # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# # appropriate corners of each of our axes, and so long as we use the
# # right transform and disable clipping.
#
# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
#
# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
#
# # What's cool about this is that now if we vary the distance between
# # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# # the diagonal lines will move accordingly, and stay right at the tips
# # of the spines they are 'breaking'
#
# # ax.get_xaxis().set_visible(False)
#
# plt.show()
# trim off bad section

# indexer = actual >-10
# expected = expected[indexer]
# actual = actual[indexer]
#
# output = np.abs((expected - actual)/expected)
# index_failures = (actual == 0)
# print("expected", expected)
# print("actual", actual)
# print("rel error", output)
# plt.scatter(expected[index_failures], output[index_failures], marker='x', label="Simulation Failed")
# plt.scatter(expected[~index_failures], output[~index_failures], label="Valid Results")
# # plt.scatter(expected, output)
# x = np.linspace(0,np.min(actual),10)
# plt.legend()
#
# # plt.plot(x,x, color='k')
# plt.xlim([expected.max()+0.1, expected.min()-0.2])
# # plt.ylim([-, actual.min()-0.2])
# plt.xlabel(r"Value of $\beta_{sim}$ used to Simulate Observations")
# plt.ylabel(r"Relative Error $\left| \frac{\beta_{sim} - \beta_{est}}{\beta_{sim}}\right|$")
# plt.show()
#
#
# indexer = (actual >-10) & (actual !=0)
#
# expected = expected[indexer]
# actual = actual[indexer]
#

output = np.abs((expected - actual)/expected)
index_failures = (actual == 0)
print("expected", expected)
print("actual", actual)
print("rel error", output)
plt.scatter(expected[index_failures], output[index_failures], marker='x', label="Simulation Failed")
plt.scatter(expected[~index_failures], output[~index_failures], label="Valid Results")
# plt.scatter(expected, output)
x = np.linspace(0,np.min(actual),10)
plt.legend()

# plt.plot(x,x, color='k')
plt.xlim([expected.max()+0.1, expected.min()-0.2])
# plt.ylim([-, actual.min()-0.2])
plt.xlabel(r"Value of $\beta_{sim}$ used to Simulate Observations")
plt.ylabel(r"Relative Error $\left| \frac{\beta_{sim} - \beta_{est}}{\beta_{sim}}\right|$")
plt.show()


# import itertools
# import time
# a = time.time()
# n =0
# for b1 in np.arange(-0.1, -2, -0.2):
#     for b2 in  np.arange(-0.01, -0.1, -0.05):
#         n+=1
#
#         beta_gen = np.array([b1, b2])
#         try:
#             obs = get_data(beta_gen, seed=2)
#         except ValueError as e:
#             print(f"beta = {beta_gen} failed, {e}")
#             continue
#         print(obs)
#         beta_init = [-0.001, -500]
#         model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
#                                               initial_beta=beta_init, mu=1,
#                                               optimiser=optimiser)
#         beta = model.solve_for_optimal_beta(verbose=True)
#         print("beta_expected", beta_gen, "beta actual", beta)
#
# b = time.time()
# print("elapsed =", b-a, "s")
