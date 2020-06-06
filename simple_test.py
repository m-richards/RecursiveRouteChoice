# Figure 2 network from fosgerau - have value function of single param cost
import numpy as np
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
print("costs")
print(costs)
utilities = -costs
beta_vec = np.array(1)

forward_incidence = np.divide(costs, costs, where=costs!=0) #np.sign(costs)
print("Incidence", )
print(forward_incidence)
mu = 1

# Need to construct utilities with parameter vec beta and then have that in such a way that we
# can optimise-  perhaps do need to consult with matlab

M = forward_incidence * np.exp(1/mu * utilities)
print("M=")
np.set_printoptions(precision=3, suppress=True)
print(M)

b_vec = np.zeros((n_nodes))
b_vec[-1] = 1

z_vec = linalg.solve(np.eye(n_nodes) -M, b_vec)



value_func = mu * np.log(z_vec)
print("Value func",)
print(value_func)
