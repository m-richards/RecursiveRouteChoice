"""Very minimal correctness checks to make sure I don't accidentally break stuff. very black box,
more intended to detect errors rather than be specific about where they are.
Also tests are not directed as to whether code is "correct" for now, they check consistency with
existing code.

"""

import pytest

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix

from data_loading import load_csv_to_sparse, load_standard_path_format_csv
from data_processing import AngleProcessor
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStruct

import os
from os.path import join
import optimisers as op


class TestSimpleCases:

    def test_first_example(self):
        subfolder = "ExampleTiny"  # big data from classical v2
        folder = join("../Datasets", subfolder)
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

        obs_mat = load_csv_to_sparse(file_obs, dtype='int', square_matrix=False).todok()

        data_list = [travel_times_mat, travel_times_mat]
        network_data_struct = RecursiveLogitDataStruct(data_list,
                                                       incidence_matrix=incidence_mat)
        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)  # TODO check these parameters & defaults

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, user_obs_mat=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6
        assert np.abs(log_like_out - 0.6931471805599454) < eps
        assert np.abs(linalg.norm(grad_out) - 0) < eps

        # model.hessian = np.identity(network_data_struct.n_dims)
        out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
        assert out_flag is True
        assert (hessian == np.identity(2)).all()
        assert optimiser.n_func_evals == 1

    # def test_basic_new_syntax(self):
    #     subfolder = "ExampleTiny"  # big data from classical v2
    #     folder = join("../Datasets", subfolder)
    #     network_data_struct, obs_mat = RecursiveLogitDataStruct.from_directory(folder,
    #                                                                            add_angles=False,
    #                                                                            delim=" ")
    #     network_data_struct.add_second_travel_time_for_testing()
    #     optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)
    #
    #     model = RecursiveLogitModel(network_data_struct, optimiser, user_obs_mat=obs_mat)
    #
    #     log_like_out, grad_out = model.get_log_likelihood()
    #     eps = 1e-6
    #     assert np.abs(log_like_out - 0.6931471805599454) < eps
    #     assert np.abs(linalg.norm(grad_out) - 0) < eps
    #
    #     # model.hessian = np.identity(network_data_struct.n_dims)
    #     out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
    #     assert out_flag == True
    #     assert (hessian == np.identity(2)).all()
    #     assert optimiser.n_func_evals == 1
    #
    def test_basic_new_new_syntax(self):
        subfolder = "ExampleTiny"  # big data from classical v2
        folder = join("../Datasets", subfolder)
        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=False)
        incidence_mat, travel_times_mat = attrs
        # left, right, _, u_turn = AngleProcessor.get_turn_categorical_matrices()
        data_list =[travel_times_mat, travel_times_mat]
        network_data_struct = RecursiveLogitDataStruct(data_list, incidence_mat)

        optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, user_obs_mat=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6
        assert np.abs(log_like_out - 0.6931471805599454) < eps
        assert np.abs(linalg.norm(grad_out) - 0) < eps

        # model.hessian = np.identity(network_data_struct.n_dims)
        out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
        assert out_flag == True
        assert (hessian == np.identity(2)).all()
        assert optimiser.n_func_evals == 1

    def test_example_tiny_modified(self):
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("../Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        data_list = [travel_times_mat, left, u_turn, t_time_incidence]
        network_data_struct = RecursiveLogitDataStruct(data_list, incidence_mat)

        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, user_obs_mat=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6

        print(optimiser.get_iteration_log(model.optim_function_state)) # Note this is currently required to
        # set the gradient so that compute relative gradient works, really bad
        # model.hessian = np.identity(network_data_struct.n_dims)
        ll, line_search_step, grad_norm, rel_grad_norm = (0.519860, 0.0, 0.467707, 0.375000)
        assert np.abs(log_like_out - ll) < eps
        assert np.abs(model.optimiser.step - line_search_step) < eps
        assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
        assert np.abs(model.optimiser.compute_relative_gradient() - rel_grad_norm) < eps

        targets = [
            (0.366296, 2.338536, 0.367698, 0.268965),
            (0.346582, 7.671693, 0.353559, 1.613793),
            (0.346582, 0.000000, 0.353559, 1.613793)
        ]
        for t in targets:
            ll, line_search_step, grad_norm, rel_grad_norm = t
            out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
            log_like_out, grad_out = model.get_log_likelihood()
            assert np.abs(log_like_out - ll) < eps
            assert np.abs(linalg.norm(model.optimiser.step) - line_search_step) < eps
            assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
            assert np.abs(model.optimiser.compute_relative_gradient() - rel_grad_norm) < eps

    def test_turn_angle_matrices(self):
        """ Note the problem of generating these kind of matrices is ignored"""
        a = np.array([[0, -0.1, 180, ],
                      [90, 0, -90],
                      [-45, -15, 0]])

        b = a * np.pi / 180
        b = csr_matrix(b)
        actual_left_turn = AngleProcessor.get_left_turn_categorical_matrix(b.todok()).toarray()
        actual_u_turn = AngleProcessor.get_u_turn_categorical_matrix(b.todok()).toarray()

        expected_left_turn = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0]])
        expected_u_turn = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        assert (expected_left_turn == actual_left_turn).all()
        assert (expected_u_turn == actual_u_turn).all()


    def test_manual_tests_dont_throw(self):
        """Just checking these scripts don't crash due to changes in api"""
        from manual_tests import (example_tiny_with_angle_data, example_tiny_no_angle_data)

