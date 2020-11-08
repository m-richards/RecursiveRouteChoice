import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from manual_verification.tntp_sioux.tntp_node_test_function import consistency_test
sns.set()

beta0 = -5
network_file = "SiouxFalls_net.tntp"
orig_indices = np.arange(0, 23, 6)
dest_indices = np.arange(1, 23, 6)
obs_per_pair = 4
obs_str = f"{len(orig_indices) * len(dest_indices) * obs_per_pair} Obs. - "

#
# expected, actual = consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0,
#                                     test_range=np.arange(-0.1, -5, -0.1))
#
#
# # second plot
# output = np.abs((expected - actual))
# index_failures = (actual == 0)
# plt.scatter(expected[~index_failures], output[~index_failures], label=obs_str, marker='+')
# plt.scatter(expected[~index_failures], output[~index_failures], label=obs_str + 'Failed Estimation')
# plt.scatter(expected, output)


# plt.ylim([-, actual.min()-0.2])

# Different data
orig_indices = np.arange(0, 23, 1)
dest_indices = np.arange(1, 23, 1)
obs_per_pair = 1
obs_str = f"{len(orig_indices) * len(dest_indices) * obs_per_pair} Obs. - "


expected, actual = consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0,
                                    test_range=np.arange(-0.1, -40, -0.5))
output = np.abs((expected - actual))
index_failures = (actual == 0)
plt.scatter(expected[~index_failures], output[~index_failures], label=obs_str, marker='+')
plt.xlabel(r"Value of $\beta_{sim}$ used to Simulate Observations")
plt.ylabel(r"Absolute Error $\left| \beta_{sim} - \beta_{est} \right|$")
plt.xlim([expected.max()+0.1, expected.min()-0.2])





plt.legend()
plt.show()
plt.savefig("ch5-beta_plot_rel4.pdf")