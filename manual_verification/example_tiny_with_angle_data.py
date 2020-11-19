# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)

import numpy as np
import time

from recursiveRouteChoice.data_loading import load_standard_path_format_csv
from recursiveRouteChoice.data_processing import AngleProcessor
from recursiveRouteChoice import RecursiveLogitModelEstimation, ModelDataStruct
from recursiveRouteChoice.optimisers import LineSearchOptimiser, OptimHessianType
import os
# np.seterr(all='raise')  # all='print')
np.set_printoptions(precision=12, suppress=True)
np.set_printoptions(edgeitems=10, linewidth=180)
np.core.arrayprint._line_width = 200
# import warnings
# warnings.simplefilter("error")

time_io_start = time.time()
# subfolder ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# subfolder = "ExampleTiny"
subfolder = "ExampleTinyModifiedObs"
print(os.getcwd())
folder = os.path.join("Datasets", subfolder)

obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
incidence_mat, travel_times_mat, angle_cts_mat = attrs
left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                      incidence_mat)
# incidence matrix which only has nonzero travel times - rather than what is specified in file
t_time_incidence = (travel_times_mat > 0).astype('int').todok()
data_list =[travel_times_mat, left, u_turn, t_time_incidence]
network_data_struct = ModelDataStruct(data_list, incidence_mat)
#
# network_data_struct, obs_mat = ModelDataStruct.from_directory(
#     folder, add_angles=True, angle_type='comparison', delim=" ")
#
# print(network_data_struct.data_array)
#
# print(network_data_struct.data_array)

time_io_end = time.time()

optimiser = LineSearchOptimiser(OptimHessianType.BFGS, max_iter=4)

model = RecursiveLogitModelEstimation(network_data_struct, optimiser, observations_record=obs_mat)
model.solve_for_optimal_beta()
time_finish = time.time()
print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}s")


print("got ", model.get_log_likelihood()[0])
print(f'{"matlab target":40}', 0.346581873680351)
print(f'{"py result (matlab data ordering )":40}', 0.3465818736803499)
print(f"{'py result (angles together ordering )':40}", 0.34658187368035076)
# 0.3465818736803499