"""Very minimal correctness checks to make sure I don't accidentally break stuff. very black box,
more intended to detect errors rather than be specific about where they are.
Also tests are not directed as to whether code is "correct" for now, they check consistency with
existing code.

"""

import pytest

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, dok_matrix

from data_loading import load_csv_to_sparse, load_standard_path_format_csv
from data_processing import AngleProcessor
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStruct, \
    RecursiveLogitModelPrediction

import os
from os.path import join
import optimisers as op

hand_net_dists = np.array(
            [[4, 3.5, 4.5, 3, 3, 0, 0, 0],
             [3.5, 3, 4, 0, 2.5, 3, 3, 0],
             [4.5, 4, 5, 0, 0, 0, 4, 3.5],
             [3, 0, 0, 2, 2, 2.5, 0, 2],
             [3, 2.5, 0, 2, 2, 2.5, 2.5, 0],
             [0, 3, 0, 2.5, 2.5, 3, 3, 2.5],
             [0, 3, 4, 0, 2.5, 3, 3, 2.5],
             [0, 0, 3.5, 2, 0, 2.5, 2.5, 2]])

hand_net_angles = np.array(
            [[180, -90, -45, 360, 90, 0, 0, 0],
             [90, 180, -135, 0, -90, -45, 360, 0],
             [45, 135, 180, 0, 0, 0, -90, 360],
             [360, 0, 0, 180, -90, 135, 0, 90],
             [-90, 90, 0, 90, 180, -135, -90, 0],
             [0, 45, 0, -135, 135, 180, -135, 135],
             [0, 360, 90, 0, 90, 135, 180, -90],
             [0, 0, 360, -90, 0, -135, 90, 180]])
hand_net_incidence = (hand_net_dists > 0).astype(int)
hand_net_angles_rad = AngleProcessor.to_radians(hand_net_angles)


class TestSimpleCases(object):

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

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, observations_record=obs_mat)

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

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, observations_record=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6
        assert np.abs(log_like_out - 0.6931471805599454) < eps
        assert np.abs(linalg.norm(grad_out) - 0) < eps

        # model.hessian = np.identity(network_data_struct.n_dims)
        out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
        assert out_flag == True
        assert (hessian == np.identity(2)).all()
        assert optimiser.n_func_evals == 1

    def test_example_tiny_modified(self): # TODO shouldn't be using this data
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

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser, observations_record=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6

        print(optimiser.get_iteration_log(model.optim_function_state)) # Note this is currently required to
        # set the gradient so that compute relative gradient works, really bad
        # model.hessian = np.identity(network_data_struct.n_dims)
        ll, line_search_step, grad_norm, rel_grad_norm = (0.519860, 0.0, 0.176776,
                                                          0.125)
        assert np.abs(log_like_out - ll) < eps
        assert np.abs(model.optimiser.step - line_search_step) < eps

        assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
        assert np.abs(model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

        targets = [
            (0.4905584, 0.1767767, 0.154794, 0.1094559),
            (0.3779969, 1.244795, 0.04175985, 0.02952867),
            (0.3634634, 0.4598828, 0.02309686, 0.01633195)
        ]
        for t in targets:
            ll, line_search_step, grad_norm, rel_grad_norm = t
            out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
            log_like_out, grad_out = model.get_log_likelihood()

            print(f"({log_like_out:.7}, "
                  f"{linalg.norm(model.optimiser.step):.7}, {linalg.norm(grad_out):.7}, "
                  f"{model.optimiser.compute_relative_gradient_non_static():.7})")

            assert np.abs(log_like_out - ll) < eps
            assert np.abs(linalg.norm(model.optimiser.step) - line_search_step) < eps
            assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
            assert np.abs(model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

    def test_example_tiny_modified_awkward_array(self):
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("../Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_record = ak.from_numpy(obs_mat)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        data_list = [travel_times_mat, left, u_turn, t_time_incidence]
        network_data_struct = RecursiveLogitDataStruct(data_list, incidence_mat)

        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser,
                                              observations_record=obs_record)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6

        print(optimiser.get_iteration_log(model.optim_function_state)) # Note this is currently required to
        # set the gradient so that compute relative gradient works, really bad
        # model.hessian = np.identity(network_data_struct.n_dims)
        ll, line_search_step, grad_norm, rel_grad_norm = (0.519860, 0.0, 0.176776,
                                                          0.125)
        assert np.abs(log_like_out - ll) < eps
        assert np.abs(model.optimiser.step - line_search_step) < eps
        assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
        assert np.abs(model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

        targets = [
            (0.4905584, 0.1767767, 0.154794, 0.1094559),
            (0.3779969, 1.244795, 0.04175985, 0.02952867),
            (0.3634634, 0.4598828, 0.02309686, 0.01633195)
        ]
        for t in targets:
            ll, line_search_step, grad_norm, rel_grad_norm = t
            out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
            log_like_out, grad_out = model.get_log_likelihood()
            assert np.abs(log_like_out - ll) < eps
            assert np.abs(linalg.norm(model.optimiser.step) - line_search_step) < eps
            assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
            assert np.abs(model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

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


class TestSimulation(object):

    def test_basic_consistency(self):
        left, right, neutral, u_turn = AngleProcessor.get_turn_categorical_matrices(dok_matrix(
            hand_net_angles_rad), dok_matrix(hand_net_incidence))

        distances = dok_matrix(hand_net_dists)

        data_list = [distances]
        network_struct = RecursiveLogitDataStruct(data_list, hand_net_incidence,
                                                  data_array_names_debug=("distances", "u_turn"))

        beta_vec = np.array([-1])

        model = RecursiveLogitModelPrediction(network_struct,
                                              initial_beta=beta_vec, mu=1)

        obs = model.generate_observations(origin_indices=[0, 1, 2, 7], dest_indices=[1, 6, 3],
                                          num_obs_per_pair=4, iter_cap=15, rng_seed=1)
        expected = [[0, 4, 5, 1, 8], [0, 1, 8], [0, 4, 1, 8], [0, 1, 8], [2, 1, 8], [2, 1, 8],
                    [2, 1, 8],
                    [2, 1, 8], [7, 7, 7, 5, 1, 8], [7, 6, 1, 8], [7, 3, 4, 1, 8], [7, 5, 1, 8],
                    [0, 1, 6, 8], [0, 1, 6, 8], [0, 1, 6, 8], [0, 3, 7, 3, 7, 6, 8], [1, 4, 6, 8],
                    [1, 6, 8], [1, 6, 8], [1, 6, 8], [2, 6, 8], [2, 6, 8], [2, 6, 8], [2, 6, 8],
                    [7, 5, 4, 6, 8], [7, 6, 8], [7, 5, 6, 8], [7, 6, 8], [0, 3, 8], [0, 3, 8],
                    [0, 3, 8],
                    [0, 3, 8], [1, 4, 3, 8], [1, 0, 3, 8], [1, 4, 3, 8], [1, 4, 3, 8], [2, 7, 3, 8],
                    [2, 7, 7, 3, 8], [2, 7, 3, 8], [2, 7, 7, 3, 8], [7, 3, 8], [7, 3, 8], [7, 3, 8],
                    [7, 3, 8]]

        assert obs ==expected

    def test_invalid_beta_throws(self):
        distances = dok_matrix(hand_net_dists)

        data_list = [distances]
        network_struct = RecursiveLogitDataStruct(data_list, hand_net_incidence,
                                                  data_array_names_debug=("distances", "u_turn"))

        beta_vec = np.array([-100])

        model = RecursiveLogitModelPrediction(network_struct,
                                              initial_beta=beta_vec, mu=1)
        try:
            model.generate_observations(origin_indices=[0, 1, 2, 7], dest_indices=[1, 6, 3],
                                          num_obs_per_pair=4, iter_cap=15, rng_seed=1)
        except ValueError as e:
            print(str(e))

class TestOptimAlgs(object):

    def test_compare_optim_methods(self):
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("../Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_record = ak.from_numpy(obs_mat)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        data_list = [travel_times_mat, left, u_turn]
        network_data_struct = RecursiveLogitDataStruct(data_list, incidence_mat)

        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser,
                                              observations_record=obs_record,
                                              initial_beta=-15)

        m1_ll_out, m1_grad_out = model.get_log_likelihood()

        optimiser2 = op.ScipyOptimiser(method='newton-cg')

        model2 = RecursiveLogitModelEstimation(network_data_struct, optimiser2,
                                              observations_record=obs_record,
                                               initial_beta=-15)
        m2_ll_out, m2_grad_out = model2.get_log_likelihood()
        eps = 1e-6

        assert np.allclose(m2_ll_out, m1_ll_out)
        assert np.allclose(m2_grad_out, m1_grad_out)

        beta1 = model.solve_for_optimal_beta()

        beta2 = model2.solve_for_optimal_beta(verbose=True)
        m1_ll_out, m1_grad_out = model.get_log_likelihood()
        m2_ll_out, m2_grad_out = model2.get_log_likelihood()
        print(m1_ll_out, m2_ll_out)
        print(m1_grad_out, m2_grad_out)

        assert np.allclose(beta1, beta2)
