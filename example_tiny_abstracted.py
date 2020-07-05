# Figure 2 network from fosgerau - have value function of single param cost
# TODO check np.dot usage since numpy is not aware of sparse properly, should use A.dot(v)

import numpy as np
import scipy

from data_loading import load_csv_to_sparse, get_incorrect_tien_turn_matrices, \
    get_uturn_categorical_matrix, get_left_turn_categorical_matrix
from main import RecursiveLogitModel, get_value_func_grad, RecursiveLogitDataSet
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix

import os
from os.path import join
import optimisers as op

# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network
# from optimisers import line_search_asrch
# from optimisers.optimisers import Optimiser

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
row, col, data = np.loadtxt(file_travel_time, unpack=True)
incidence_data = np.ones(len(data))


travel_times_mat = load_csv_to_sparse(file_travel_time)
incidence_mat = load_csv_to_sparse(file_incidence, dtype='int')
turn_angle_mat = load_csv_to_sparse(file_turn_angle)

# turn turn angle data into uturn and left turn dummies
print(turn_angle_mat.toarray())


tien_left_turn = incidence_mat
tien_actual_left_dummy, tien_uturn_dummy = get_incorrect_tien_turn_matrices(
    turn_angle_mat)

left_turn_dummy = get_left_turn_categorical_matrix(turn_angle_mat)
u_turn_dummy = get_uturn_categorical_matrix(turn_angle_mat)

#
#
# # Get observations matrix - note: observation matrix is in sparse format, but is of the form
# #   each row == [dest node, orig node, node 2, node 3, ... dest node, 0 padding ....]
# obs_mat = load_csv_to_sparse(file_obs, dtype='int', square_matrix=False)
#
# network_data_struct = RecursiveLogitDataSet(travel_times=travel_times_mat,
#                                             incidence_matrix=incidence_mat,
#                                             turn_angles=None)
# optimiser = op.LineSearchOptimiser(op.OptimType.LINE_SEARCH, op.OptimHessianType.BFGS,
#                                    vec_length=1,
#                                    max_iter=4)  # TODO check these parameters & defaults
#
# model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat=obs_mat)
# np.set_printoptions(precision=4, suppress=True)
# np.set_printoptions(edgeitems=3)
# np.core.arrayprint._line_width = 100
#
# # from optimisers import log_likelihood
# beta = np.array(-1.5)  # default value, 1d for now
# # log_likelihood(beta, data_struct, obs_mat)
# #
# beta_vec = beta
# data = network_data_struct
# obs = obs_mat
#
# # temp func
# # def log_likelihood(beta_vec, data:DataSet, obs, mu=1):
#
#
# log_like_out, grad_out = model.get_log_likelihood()
#
# print("Target:\nLL =0.6931471805599454\ngrad=[0.]")
# print("Got:")
# print("Got LL = ", log_like_out)
# print("got grad_cumulative = ", grad_out)
#
# model.hessian = hessian = np.identity(data.n_dims)
# print(optimiser.line_search_iteration(model))
