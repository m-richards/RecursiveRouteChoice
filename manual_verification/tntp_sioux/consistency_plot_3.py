import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from brokenaxes import brokenaxes
from manual_verification.tntp_sioux.tntp_node_test_function import consistency_test
sns.set()

beta0 = -5
max_nodep1 = 23 + 1
network_file = "SiouxFalls_net.tntp"
orig_indices = np.arange(0, 24, 6)
dest_indices = (orig_indices + 5) % 24
print("first dest indices are", dest_indices)
obs_per_pair = 4
obs_str1 = f"{len(orig_indices) * len(dest_indices) * obs_per_pair} Obs."
import sys
sys.stdout = open("3samples_obs.txt", "w")
#
# test_range = np.arange(-0.1, -10, -0.6)
test_range = np.r_[np.arange(-0.1, -4, -0.3), np.arange(-4, -10, -0.6)]
expected1, actual1 = consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0,
                                      test_range=test_range)

# Different data
# orig_indices = np.arange(0, 23, 1)
orig_indices = np.arange(0, max_nodep1, 1)
# dest_indices = np.arange(1, 23, 1)
dest_indices = np.arange(0, max_nodep1, 1)
obs_per_pair = 1
obs_str2 = f"{len(orig_indices) * len(dest_indices) * obs_per_pair} Obs."

expected2, actual2 = consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0,
                                    test_range=test_range)

# Different data

# orig_indices = np.arange(0, 23, 1)
orig_indices3 = np.arange(0, max_nodep1, 1)
# dest_indices = np.arange(1, 23, 1)
dest_indices3 = np.arange(1, max_nodep1, 1)
obs_per_pair3 = 2
obs_str3 = f"{len(orig_indices3) * len(dest_indices3) * obs_per_pair3} Obs."
expected3, actual3 = consistency_test(network_file, orig_indices3, dest_indices3, obs_per_pair3,
                                      beta0,
                                    test_range=test_range)

# # second plot
output1 = np.abs((expected1 - actual1))
index_failures1 = (actual1 == 0)
output2 = np.abs((expected2 - actual2))
index_failures2 = (actual2 == 0)
output3 = np.abs((expected3 - actual3))
index_failures3 = (actual3 == 0)
plt.figure(figsize=(8,6))
plt.scatter(expected1[index_failures1], output1[index_failures1], label='Simulation '
                                                                                   'Failed',
            marker='x')
plt.scatter(expected1[~index_failures1], output1[~index_failures1], label=obs_str1, marker='1')
plt.scatter(expected2[~index_failures2], output2[~index_failures2], label=obs_str2, marker='2')
plt.scatter(expected3[~index_failures3], output3[~index_failures3], label=obs_str3, marker='3')
# plt.scatter(expected2[index_failures2], output2[index_failures2], label=obs_str2+ 'Failed '
#                                                                                     'Estimation',
#                                                                                     marker='x')
plt.xlabel(r"Value of $\beta_{sim}$ used to Simulate Observations")
plt.ylabel(r"Absolute Error $\left| \beta_{sim} - \beta_{est} \right|$")
plt.xlim([expected2.max()+0.1, expected2.min()-0.2])
print("yvals\n", output1, "\n", output2)

plt.legend()
# plt.show()
plt.savefig("ch5-beta_plot_relv2.5.pdf")
plt.figure(figsize=(8,6))
plt.scatter(expected1[index_failures1], output1[index_failures1], label='Simulation '
                                                                                   'Failed',
            marker='x')
plt.scatter(expected1[~index_failures1], output1[~index_failures1], label=obs_str1, marker='1')
plt.scatter(expected2[~index_failures2], output2[~index_failures2], label=obs_str2, marker='2')
plt.scatter(expected3[~index_failures3], output3[~index_failures3], label=obs_str3, marker='3')
# plt.scatter(expected2[index_failures2], output2[index_failures2], label=obs_str2+ 'Failed '
#                                                                                     'Estimation',
#                                                                                     marker='x')
plt.xlabel(r"Value of $\beta_{sim}$ used to Simulate Observations")
plt.ylabel(r"Absolute Error $\left| \beta_{sim} - \beta_{est} \right|$")
plt.xlim([expected2.max()+0.1, -5.0])
plt.ylim([-0.1, 2])
print("yvals\n", output1, "\n", output2)
sys.stdout.close()
plt.legend()
# plt.show()
plt.savefig("ch5-beta_plot_relv2.5a.pdf")




# broken axis plot


ylim = [.05, -5.75]
ylim2 = [-13.25, -18]
bax = brokenaxes(ylims=(ylim, ylim2), hspace=.05, fig=plt.figure(figsize=(8,6)))
bax.scatter(expected1[index_failures1], actual1[index_failures1], marker='x',
            label="Simulation Failed")
bax.scatter(expected1[~index_failures1], actual1[~index_failures1], label=obs_str1, marker='1')
bax.scatter(expected2[~index_failures2], actual2[~index_failures2], label=obs_str2, marker='2')
bax.scatter(expected3[~index_failures3], actual3[~index_failures3], label=obs_str3, marker='3')
x = np.linspace(0, np.min(actual1), 10)
bax.plot(x, x, color='k')
bax.set_xlim([0, -10.1])
bax.set_xlabel(r"Value of $\beta$ used to Simulate")
bax.set_ylabel(r"Value of $\beta$ obtained from Estimation")
bax.legend(loc='upper left')
# plt.show()

plt.savefig("ch5-beta_plot_linv2.5.pdf")


