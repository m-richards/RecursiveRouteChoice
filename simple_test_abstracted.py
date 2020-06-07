# Figure 2 network from fosgerau - have value function of single param cost
import numpy as np
from main import DataSet
from scipy import linalg
# from scipy.linalg import expm
END = 'end'
END_VAL = 0
n_nodes = 5 # including end
# Node to node costs, not arc to arc
costs_dict = {(1, 2)  : 1,
         (1, 5)  : 4,
         (2, 3)  : 1,
         (2, 4)  : 2,
         (3, 4)  : 1,
         (3, END)  : 2,
         (4, END)  : 1,
         # (5, END): END_VAL,
         }
# in general, we would have N*N*n_value_func_params costs array
costs = np.zeros((n_nodes, n_nodes))
for ((i, j), v) in costs_dict.items():
    if j == END:
        j = n_nodes
    costs[i - 1, j - 1] = v
# print("costs")
# print(costs)
utilities = -costs
beta_vec = np.array(1)

forward_incidence = np.divide(costs, costs, where=costs!=0) #np.sign(costs)
# Note that we maximise negative travel time
data = DataSet(travel_times=costs, incidence_matrix=forward_incidence, turn_angles=None)