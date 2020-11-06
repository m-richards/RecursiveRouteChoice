import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from manual_verification.tntp_sioux.tntp_node_test_function import consistency_test
sns.set()
beta0 = -5
network_file = "SiouxFalls_net.tntp"
orig_indices = np.arange(0, 23, 6)
dest_indices = np.arange(1, 23, 6)
obs_per_pair = 1
expected, actual = consistency_test(network_file, orig_indices, dest_indices, obs_per_pair, beta0)
actual = np.array(actual)
expected = np.array(expected)
print(list(expected))
print(list(actual))

from brokenaxes import brokenaxes

ylim = [0, -5]
ylim2 = [-6, -14]
indexer = actual <-0.1

bax = brokenaxes(ylims=(ylim, ylim2), hspace=.05)
obs_str = "24 Obs. - "
bax.scatter(expected[indexer], actual[indexer], label=obs_str + 'Successful Estimation')
bax.scatter(expected[~indexer], actual[~indexer], marker='x', label=obs_str + 'Failed Estimation')
x = np.linspace(0, np.min(actual), 10)
bax.plot(x, x, color='k')
bax.set_xlim([0, -2.1])
bax.set_xlabel(r"$\beta{sim}$ used to generate Obs")
bax.set_ylabel(r"$\beta{est}$ Recovered from Estimation")
plt.legend()
plt.savefig("ch5-beta_plot.pdf")
plt.figure()
