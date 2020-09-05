import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix, coo_matrix, load_npz
from scipy.sparse import linalg as splinalg

from debug_helpers import print_sparse
np.set_printoptions(precision=12, linewidth=220)
import os
print(os.getcwd())
# base = "case_2"
# from os.path import join
# a_sp =load_npz(join(base, "broken_a_mat_sparse.npz"))
# rhs_sp = load_npz((join(base, "broken_rhs_sparse.npz")))
#
# a_de = np.load(join(base, "broken_a_mat_dense.npy"), allow_pickle=False)
# rhs_de = np.load(join(base, "broken_rhs_dense.npy"), allow_pickle=False)
# print_sparse(a_sp)
# print(a_de)
# z_vec_sp = splinalg.spsolve(a_sp, rhs_sp)
# z_vec_sp = np.atleast_2d(z_vec_sp).T
# print("z_sp = ")
# print(z_vec_sp.T)
# print("res vec sp", a_sp.shape, z_vec_sp.shape, rhs_sp.shape)
# res = a_sp * z_vec_sp - rhs_sp
# print(res.T)
# print("res norm = ", linalg.norm(np.array(res)))
#
# z_vec_de = linalg.solve(a_de, rhs_de)
# print("DENSE CASE")
# # z_vec_de = np.atleast_2d(z_vec_de).T
# print("z_de = ")
# print(z_vec_de.T)
# print("res vec de", a_de.shape, z_vec_de.shape, rhs_de.shape)
# res = a_de @ z_vec_de - rhs_de
# print(res.T)
# print("res norm = ", linalg.norm(np.array(res)))
#
# #
# #
# print("A condition number", np.linalg.cond(a_sp.toarray(), p=2))
#
# # print(np.linalg.inv(a_de) @a_de)
#
# print("z_diff", z_vec_de - z_vec_sp)
# # Different solutions for sparse and dense, one has appropriate res, other is wrong
#
# print("a equal", np.all(a_de==a_sp.toarray()), "rhs_equal", np.all(rhs_sp.toarray()==rhs_de))
#
# print(linalg.solve(a_de, rhs_de).T.squeeze())
# print(splinalg.spsolve(a_sp, rhs_sp))
# print(linalg.solve(a_sp.toarray(), rhs_sp.toarray()).T.squeeze())
# print(rhs_de.T)

nr = 9
nc = 9
rhs = np.zeros((9,1))
rhs[-1,0] = 1
cond_num = 1  # %Desired condition number

def get_average_norm_sparse_dense_diff(cond_num, reps=5):
    diffs = []
    residuals_sp = []
    residuals_de = []

    for i in range(reps):
        A = np.random.random((nr, nc))
        # print(np.linalg.cond(A))
        U, s, Vh = linalg.svd(A, full_matrices=False)
        S = np.diag(s)
        S[S != 0] = np.linspace(cond_num, 1, min(nr, nc))
        # Modified condition number matrix
        A2 = np.dot(U, np.dot(S, Vh))

        # print("{0:.12e}".format(float(np.linalg.cond(A2))))
        # print(rhs)

        z_dense = linalg.solve(A2, rhs)
        z_sparse = np.atleast_2d(splinalg.spsolve(csr_matrix(A2), csr_matrix(rhs))).T
        # print(z_sparse)
        # print(z_dense.T.squeeze())
        # print(np.allclose(z_dense.squeeze(), z_sparse))
        # print(z_sparse.shape, z_dense.shape)
        # print((z_sparse-z_dense).T)
        diffs.append(linalg.norm(z_sparse - z_dense))
        # print(csr_matrix(A2).shape, z_sparse.shape, csr_matrix(rhs).shape)
        res_sp = csr_matrix(A2) @ z_sparse - csr_matrix(rhs)
        res_de = A2 @ z_dense - rhs
        residuals_sp.append(linalg.norm(res_sp))
        residuals_de.append(linalg.norm(res_de))
    return diffs, residuals_sp, residuals_de
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
cond_nums = np.logspace(0,25,26)
print("cond numbs", cond_nums)
fig, axs = plt.subplots(1,3, figsize=(10,7), sharex=False)

# plt.xticks()
MEAN = True
for c in cond_nums:
    sparse_dense_diff_norm, res_sp, res_de = get_average_norm_sparse_dense_diff(c,reps=10)
    print("\t", c, sparse_dense_diff_norm)
    [i.set_prop_cycle(None) for i in axs]
    if MEAN:
        for ax, n in zip(axs, (np.mean(sparse_dense_diff_norm),
                               np.mean(res_sp),
                               np.mean(res_de))):
            plt.axes(ax)
            plt.scatter(c, n, figure=fig)
    else:

        for (i,j,k) in zip(sparse_dense_diff_norm, res_sp, res_de):
            for ax, n in zip(axs, (i,j,k)):
                plt.axes(ax)
                plt.scatter(c, n, figure=fig)


for ax in axs:
    plt.axes(ax)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Average matrix condition number")
    plt.ylim([1e-12,1e3])
plt.axes(axs[0])
plt.ylabel("Sp. Dense Diff Norm")#("Norm of difference between sparse and dense soln")
plt.axes(axs[1])
plt.ylabel("Sp res")#("solution residual for sparse solution")
plt.axes(axs[2])
plt.ylabel("De res")#("solution residual for dense solution")




plt.show()




