import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import linalg
import seaborn as sns
from scipy.linalg import lu_factor, lu_solve




sns.set()
from numpy.random import default_rng
rng =default_rng()
import time

xticks = np.logspace(6, 11, base=2, num=6)
print(xticks)
yticks = np.logspace(-7, 9, base=2, num=17)
print(yticks)
# N = np.arange(10,300,10)
start = time.time()
a = 1
b = np.log10(1000)
# N = np.logspace(a,b, endpoint=True, num=2).astype(int)
# print(N)
N = np.array([100,200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
direct = np.zeros(len(N), float)
direct_explicit_inv = np.zeros(len(N), float)
sm_inverse = np.zeros(len(N), float)
sm_lu = np.zeros(len(N), float)
sm_lu_scipy = np.zeros(len(N), float)
AVERAGE_OVER = 5
# fraction_of_dest_covered = 0.9
# for i, n in enumerate(N):
#     print(n)
#     n = int(n)
#     for j in range(AVERAGE_OVER):
#         A = rng.random((n, n))  # note A [0,1) is appropriate given definition of M, could include
#         # 1 also, but probability is neglible
#         A = 1000 * A
#         v = np.zeros((n,1))
#         v[-1] = 1
#         b = v.copy()
#
#         # SM with inverse explicitly forms Ainv
#         t2s = time.time()
#         Ainv = linalg.inv(A, check_finite=True)
#         Ainv_b = Ainv @ b
#         t2d = time.time() - t2s
#         sm_inverse[i] += t2d
#
#         # LU sherman morrison
#         t5s = time.time()
#         # lu_obj = DenseLUObj(A)
#         A_lu, A_piv = lu_factor(A, check_finite=True)
#         lu_Ainv_b = lu_solve((A_lu, A_piv), b, check_finite=True).reshape(n, 1)
#         t5d = time.time() - t5s
#         sm_lu[i] +=t5d
#
#         # other LU in scipy
#         t7s = time.time()
#         AP, AL, AU = linalg.lu(A, check_finite=True)
#         b_tilde = AP.T @ b
#         y = linalg.solve_triangular(AL, b_tilde, lower=True, check_finite=True)
#         lu_Ainv_b2 = linalg.solve_triangular(AU, y, check_finite=True)
#         t7d = time.time() - t7s
#         sm_lu_scipy[i] += t7d
#
#
#
#
#         for obs in range(int(fraction_of_dest_covered * n)):
#             # u for this "dest"
#             u = rng.integers(0, 1, (n, 1), endpoint=True)
#             Aprime = A + u @ v.T
#             # direct solve
#             t1s = time.time()
#             z1 = linalg.solve(Aprime, b, check_finite=True)
#             t1d = time.time() - t1s
#             direct[i] += t1d
#             # explicit inverse formation, should be bad
#             t3s = time.time()
#             z2 = linalg.inv(Aprime, check_finite=True) @ b
#             t3d = time.time() - t3s
#             direct_explicit_inv[i] += t3d
#
#             # direct SM
#             t4s = time.time()
#             Ainv_u = Ainv @ u
#             z3 = Ainv_b - Ainv_u * (v.T @ Ainv_b) / (1 + v.T @ Ainv_u)
#             t4d = time.time() - t4s
#             sm_inverse[i] += t4d
#
#             # SM with LU reuse
#             t6s = time.time()
#             lu_Ainv_u = lu_solve((A_lu, A_piv), u, check_finite=True)
#             z4 = lu_Ainv_b - lu_Ainv_u * v.T @ lu_Ainv_b / (1 + v.T @ lu_Ainv_u)
#             t6d = time.time() - t6s
#             sm_lu[i] += t6d
#
#             # SM with LU reuse scipy (far slower)
#
#             t8s = time.time()
#             b_tilde = AP.T @ u
#             y = linalg.solve_triangular(AL, b_tilde, lower=True, check_finite=True)
#             lu_Ainv_u = linalg.solve_triangular(AU, y, check_finite=True)
#             z5 = lu_Ainv_b - lu_Ainv_u * v.T @ lu_Ainv_b / (1 + v.T @ lu_Ainv_u)
#             t8d = time.time() - t8s
#             sm_lu_scipy[i] += t8d
#
#             assert np.allclose(z1, z2)
#             assert np.allclose(z1, z3)
#             assert np.allclose(z1, z4)
#             assert np.allclose(z1, z5)
#         print("Direct inv", direct_explicit_inv[i])
#         print("Direct LS", direct[i])
#         print("SM inv", sm_inverse[i])
#         print("SM lu", sm_lu[i])
#         print("SM lu not lu factor", sm_lu_scipy[i])
#
# direct_explicit_inv /= AVERAGE_OVER
# sm_inverse /= AVERAGE_OVER
# sm_lu /= AVERAGE_OVER
# direct /= AVERAGE_OVER
#
# plt.figure(figsize=(8,6))
# plt.scatter(N, direct_explicit_inv, marker='1', label="Direct - Explicit Inv.")
# plt.scatter(N, direct, marker='2', label="Direct - Linear Sys.")
# plt.scatter(N, sm_inverse, marker='3', label="SM - Explicit Inv.")
# plt.scatter(N, sm_lu, marker='4', label="SM - Linear Sys. with LU (LAPACK)")
# plt.scatter(N, sm_lu_scipy, marker='4', label="SM - Linear Sys. with LU (SciPy C)")
# plt.legend()
# plt.xlabel("Matrix system side length N")
# plt.ylabel(f"Runtime Averaged Over {AVERAGE_OVER} Runs (s)")
# plt.xscale("log")
# plt.yscale("log")
#
# plt.xticks(xticks)
# plt.yticks(yticks)
# plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# # plt.show()
#
# plt.savefig("with_checks1.pdf")
#
# end = time.time()
# print("tot runtime", end-start)
# print("dir explicit\n", direct_explicit_inv)
# print("dir linear sys\n", direct)
# print("SM inv\n", sm_inverse)
# print("SM lu_factor (LAPACK)\n", sm_lu)
# print("SM lu \n", sm_lu_scipy)
#
#
# print("tot runtime", end-start)
# print("dir explicit\n", list(direct_explicit_inv))
# print("dir linear sys\n", list(direct))
# print("SM inv\n", list(sm_inverse))
# print("SM lu_factor (LAPACK)\n", list(sm_lu))
# print("SM lu \n", list(sm_lu_scipy))

#########################################################
#
# direct = np.zeros(len(N), float)
# direct_explicit_inv = np.zeros(len(N), float)
# sm_inverse = np.zeros(len(N), float)
# sm_lu = np.zeros(len(N), float)
# sm_lu_scipy = np.zeros(len(N), float)
# AVERAGE_OVER = 1
fraction_of_dest_covered = 0.9
for i, n in enumerate(N):
    print(n)
    n = int(n)
    for j in range(AVERAGE_OVER):
        A = rng.random((n, n))
        v = np.zeros((n,1))
        v[-1] = 1
        b = v.copy()

        # SM with inverse explicitly forms Ainv
        t2s = time.time()
        Ainv = linalg.inv(A, check_finite=False)
        Ainv_b = Ainv @ b
        t2d = time.time() - t2s
        sm_inverse[i] += t2d

        # LU sherman morrison
        t5s = time.time()
        # lu_obj = DenseLUObj(A)
        A_lu, A_piv = lu_factor(A, check_finite=False)
        lu_Ainv_b = lu_solve((A_lu, A_piv), b, check_finite=False).reshape(n, 1)
        t5d = time.time() - t5s
        sm_lu[i] +=t5d

        # other LU in scipy
        t7s = time.time()
        AP, AL, AU = linalg.lu(A, check_finite=False)
        b_tilde = AP.T @ b
        y = linalg.solve_triangular(AL, b_tilde, lower=True, check_finite=False)
        lu_Ainv_b2 = linalg.solve_triangular(AU, y, check_finite=False)
        t7d = time.time() - t7s
        sm_lu_scipy[i] += t7d




        for obs in range(int(fraction_of_dest_covered * n)):
            # u for this "dest"
            u = rng.integers(0, 1, (n, 1), endpoint=True)
            Aprime = A + u @ v.T
            # direct solve
            t1s = time.time()
            z1 = linalg.solve(Aprime, b, check_finite=False)
            t1d = time.time() - t1s
            direct[i] += t1d
            # explicit inverse formation, should be bad
            t3s = time.time()
            z2 = linalg.inv(Aprime, check_finite=False) @ b
            t3d = time.time() - t3s
            direct_explicit_inv[i] += t3d

            # direct SM
            t4s = time.time()
            Ainv_u = Ainv @ u
            z3 = Ainv_b - Ainv_u * (v.T @ Ainv_b) / (1 + v.T @ Ainv_u)
            t4d = time.time() - t4s
            sm_inverse[i] += t4d

            # SM with LU reuse
            t6s = time.time()
            lu_Ainv_u = lu_solve((A_lu, A_piv), u, check_finite=False)
            z4 = lu_Ainv_b - lu_Ainv_u * v.T @ lu_Ainv_b / (1 + v.T @ lu_Ainv_u)
            t6d = time.time() - t6s
            sm_lu[i] += t6d

            # SM with LU reuse scipy (far slower)

            t8s = time.time()
            b_tilde = AP.T @ u
            y = linalg.solve_triangular(AL, b_tilde, lower=True, check_finite=False)
            lu_Ainv_u = linalg.solve_triangular(AU, y, check_finite=False)
            z5 = lu_Ainv_b - lu_Ainv_u * v.T @ lu_Ainv_b / (1 + v.T @ lu_Ainv_u)
            t8d = time.time() - t8s
            sm_lu_scipy[i] += t8d

            assert np.allclose(z1, z2)
            assert np.allclose(z1, z3)
            assert np.allclose(z1, z4)
            assert np.allclose(z1, z5)
        print("Direct inv", direct_explicit_inv[i])
        print("Direct LS", direct[i])
        print("SM inv", sm_inverse[i])
        print("SM lu", sm_lu[i])
        print("SM lu not lu factor", sm_lu_scipy[i])

direct_explicit_inv /= AVERAGE_OVER
sm_inverse /= AVERAGE_OVER
sm_lu /= AVERAGE_OVER
direct /= AVERAGE_OVER

plt.figure(figsize=(8,6))
plt.scatter(N, direct_explicit_inv, marker='1', label="Direct - Explicit Inv.")
plt.scatter(N, direct, marker='2', label="Direct - Linear Sys.")
plt.scatter(N, sm_inverse, marker='3', label="SM - Explicit Inv.")
plt.scatter(N, sm_lu, marker='4', label="SM - Linear Sys. with LU (LAPACK)")
plt.scatter(N, sm_lu_scipy, marker='4', label="SM - Linear Sys. with LU (SciPy C)")
plt.legend()
plt.xlabel("Matrix system side length N")
plt.ylabel(f"Runtime Averaged Over {AVERAGE_OVER} Runs (s)")
plt.xscale("log")
plt.yscale("log")

plt.xticks(xticks)
plt.yticks(yticks)
plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.show()

plt.savefig("nocheck2.pdf")


end = time.time()
import sys
sys.stdout = open("test.txt", "w")





print("tot runtime", end-start)
print("dir explicit\n", direct_explicit_inv)
print("dir linear sys\n", direct)
print("SM inv\n", sm_inverse)
print("SM lu_factor (LAPACK)\n", sm_lu)
print("SM lu \n", sm_lu_scipy)


print("tot runtime", end-start)
print("dir explicit\n", list(direct_explicit_inv))
print("dir linear sys\n", list(direct))
print("SM inv\n", list(sm_inverse))
print("SM lu_factor (LAPACK)\n", list(sm_lu))
print("SM lu \n", list(sm_lu_scipy))

sys.stdout.close()


import os
os.system('shutdown /p /f')