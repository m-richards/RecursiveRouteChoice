# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import time
import numpy as np

from main import RecursiveLogitModel, RecursiveLogitDataStruct

from os.path import join
import optimisers as op
from optimisers import OptimType

np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(edgeitems=3)
np.seterr(all='raise')  # all='print')
import warnings

warnings.simplefilter("error")

# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network

time_io_start = time.time()
subfolder = "ExampleTiny"  # big data from classical v2
folder = join("../Datasets", subfolder)

time_io_end = time.time()
network_data_struct, obs_mat = RecursiveLogitDataStruct.from_directory(folder, add_angles=False,
                                                                       delim=" ")
network_data_struct.add_second_travel_time_for_testing()
optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS,
                                   vec_length=1,
                                   max_iter=4)  # TODO check these parameters & defaults

model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat=obs_mat)

model.solve_for_optimal_beta()

time_finish = time.time()
# tODO covariance
print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}")
