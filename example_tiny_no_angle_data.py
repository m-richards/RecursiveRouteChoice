# Figure 2 network from fosgerau - have value function of single param cost
# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)
import time
import numpy as np
import scipy
from data_loading import load_csv_to_sparse, get_incorrect_tien_turn_matrices, \
    get_uturn_categorical_matrix, get_left_turn_categorical_matrix
from main import RecursiveLogitModel, get_value_func_grad, RecursiveLogitDataSet

import os
from os.path import join
import optimisers as op
from optimisers import OptimType

np.seterr(all='raise')  # all='print')
import warnings

warnings.simplefilter("error")

# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network

time_io_start = time.time()
subfolder = "ExampleTiny"  # big data from classical v2
folder = join("Datasets", subfolder)
INCIDENCE = "incidence.txt"
TRAVEL_TIME = 'travelTime.txt'
OBSERVATIONS = "observations.txt"
TURN_ANGLE = "turnAngle.txt"
file_incidence = os.path.join(folder, INCIDENCE)
file_travel_time = os.path.join(folder, TRAVEL_TIME)
file_turn_angle = os.path.join(folder, TURN_ANGLE)
file_obs = os.path.join(folder, OBSERVATIONS)

travel_times_mat = load_csv_to_sparse(file_travel_time).todok()
incidence_mat = load_csv_to_sparse(file_incidence, dtype='int').todok()
turn_angle_mat = load_csv_to_sparse(file_turn_angle).todok()

tien_left_turn = incidence_mat
tien_actual_left_dummy, tien_uturn_dummy = get_incorrect_tien_turn_matrices(
    turn_angle_mat)

left_turn_dummy = get_left_turn_categorical_matrix(turn_angle_mat)
u_turn_dummy = get_uturn_categorical_matrix(turn_angle_mat)

# Get observations matrix - note: observation matrix is in sparse format, but is of the form
#   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
obs_mat = load_csv_to_sparse(file_obs, dtype='int', square_matrix=False).todok()
time_io_end = time.time()
network_data_struct = RecursiveLogitDataSet(travel_times=travel_times_mat,
                                            incidence_matrix=incidence_mat,
                                            turn_angles=None)
optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS,
                                   vec_length=1,
                                   max_iter=4)  # TODO check these parameters & defaults

model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat=obs_mat)
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 100

log_like_out, grad_out = model.get_log_likelihood()

model.hessian = np.identity(network_data_struct.n_dims)
n = 0
print("Initial Values:")
optimiser.set_beta_vec(model.beta_vec)
optimiser.set_current_value(log_like_out)
print(optimiser._line_search_iteration_log(model))
while n <= 1000:
    optimiser.iter_count += 1
    if model.optimiser.METHOD_FLAG == OptimType.LINE_SEARCH:
        ok_flag, hessian, log_msg = optimiser.line_search_iteration(model, verbose=False)
        if ok_flag:
            print(log_msg)
        else:
            raise ValueError("Line search error flag was raised. Process failed.")
    else:
        raise NotImplementedError("Only have line search implemented")

    # check stopping condition
    is_stopping, stop_type, is_successful = optimiser.check_stopping_criteria()

    if is_stopping:
        print(f"The algorithm stopped due to condition: {stop_type}")
        break

if n == 1000:
    print("Infinite loop happened somehow, shouldn't have happened")

time_finish = time.time()
# tODO covariance
print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}")
