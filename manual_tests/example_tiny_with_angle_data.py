# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import os

import numpy as np
import time

from main import RecursiveLogitModel, RecursiveLogitDataStruct
from optimisers import LineSearchOptimiser, OptimHessianType

np.seterr(all='raise')  # all='print')
np.set_printoptions(precision=6, suppress=True)
np.set_printoptions(edgeitems=10, linewidth=180)
np.core.arrayprint._line_width = 200
# import warnings
# warnings.simplefilter("error")

time_io_start = time.time()
# subfolder ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# subfolder = "ExampleTiny"
subfolder = "ExampleTinyModifiedObs"
folder = os.path.join("../Datasets", subfolder)

network_data_struct, obs_mat = RecursiveLogitDataStruct.from_directory(
    folder, add_angles=True, angle_type='comparison', delim=" ")

time_io_end = time.time()

optimiser = LineSearchOptimiser(OptimHessianType.BFGS, max_iter=4)

model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat=obs_mat)
model.solve_for_optimal_beta()
time_finish = time.time()
# tODO covariance
print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}s")

print("want", 0.34658187368035076)
print("got ", model.get_log_likelihood()[0])