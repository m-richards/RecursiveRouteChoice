import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, coo_matrix, load_npz
from scipy.sparse import linalg as splinalg

from debug_helpers import print_sparse
np.set_printoptions(precision=12, linewidth=220)
import os
print(os.getcwd())
base = "case_2"
from os.path import join
a_sp =load_npz(join(base, "broken_a_mat_sparse.npz"))
rhs_sp = load_npz((join(base, "broken_rhs_sparse.npz")))

a_de = np.load(join(base, "broken_a_mat_dense.npy"), allow_pickle=False)
rhs_de = np.load(join(base, "broken_rhs_dense.npy"), allow_pickle=False)
print_sparse(a_sp)
print(a_de)
z_vec_sp = splinalg.spsolve(a_sp, rhs_sp)
z_vec_sp = np.atleast_2d(z_vec_sp).T
print("z_sp = ")
print(z_vec_sp.T)
print("res vec sp", a_sp.shape, z_vec_sp.shape, rhs_sp.shape)
res = a_sp @ z_vec_sp - rhs_sp
print(res.T)
print("res norm = ", linalg.norm(np.array(res)))

z_vec_de = linalg.solve(a_de, rhs_de)
print("DENSE CASE")
# z_vec_de = np.atleast_2d(z_vec_de).T
print("z_de = ")
print(z_vec_de.T)
print("res vec de", a_de.shape, z_vec_de.shape, rhs_de.shape)
res = a_de @ z_vec_de - rhs_de
print(res.T)
print("res norm = ", linalg.norm(np.array(res)))

#
#
print("A condition number", np.linalg.cond(a_sp.toarray(), p=2))

# print(np.linalg.inv(a_de) @a_de)

print("z_diff", z_vec_de - z_vec_sp)
# Different solutions for sparse and dense, one has appropriate res, other is wrong

print("a equal", np.all(a_de==a_sp.toarray()), "rhs_equal", np.all(rhs_sp.toarray()==rhs_de))

print(linalg.solve(a_de, rhs_de).T.squeeze())
print(splinalg.spsolve(a_sp, rhs_sp))
print(linalg.solve(a_sp.toarray(), rhs_sp.toarray()).T.squeeze())
print(rhs_de.T)



