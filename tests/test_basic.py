"""Very minimal correctness checks to make sure I don't accidentally break stuff. very black box,
more intended to detect errors rather than be specific about where they are.
Also tests are not directed as to whether code is "correct" for now, they check consistency with
existing code.

"""
from recursiveRouteChoice.recursive_route_choice import ALLOW_POSITIVE_VALUE_FUNCTIONS
# import pytest

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, dok_matrix, issparse

from recursiveRouteChoice.data_loading import load_csv_to_sparse, load_standard_path_format_csv
from recursiveRouteChoice.data_processing import AngleProcessor
from recursiveRouteChoice import RecursiveLogitModelEstimation, ModelDataStruct, \
    RecursiveLogitModelPrediction
from recursiveRouteChoice import optimisers
import os
from os.path import join

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

    @staticmethod
    def _first_example_common_data_checks(travel_times_mat, incidence_mat, obs_mat):
        data_list = [travel_times_mat, travel_times_mat]
        network_data_struct = ModelDataStruct(data_list, incidence_mat)

        optimiser = optimisers.LineSearchOptimiser(optimisers.OptimHessianType.BFGS, max_iter=4)

        model = RecursiveLogitModelEstimation(network_data_struct, optimiser,
                                              observations_record=obs_mat)

        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6
        assert np.abs(log_like_out - 2.7725887222397816) < eps
        assert np.abs(linalg.norm(grad_out) - 0) < eps

        # model.hessian = np.identity(network_data_struct.n_dims)
        out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state, verbose=False)
        assert out_flag is True
        assert (hessian == np.identity(2)).all()
        assert optimiser.n_func_evals == 1

    @staticmethod
    def load_example_tiny_manually():
        subfolder = "ExampleTiny"  # big data from classical v2
        folder = join("Datasets", subfolder)
        INCIDENCE = "incidence.txt"
        TRAVEL_TIME = 'travelTime.txt'
        OBSERVATIONS = "observations.txt"
        # TURN_ANGLE = "turnAngle.txt"
        file_incidence = os.path.join(folder, INCIDENCE)
        file_travel_time = os.path.join(folder, TRAVEL_TIME)
        # file_turn_angle = os.path.join(folder, TURN_ANGLE)
        file_obs = os.path.join(folder, OBSERVATIONS)

        travel_times_mat = load_csv_to_sparse(file_travel_time).todok()
        incidence_mat = load_csv_to_sparse(file_incidence, dtype='int').todok()

        obs_mat = load_csv_to_sparse(file_obs, dtype='int', square_matrix=False).todok()
        return travel_times_mat, incidence_mat, obs_mat

    def test_example_manual_loading_sparse_raw(self):
        travel_times_mat, incidence_mat, obs_mat = self.load_example_tiny_manually()
        self._first_example_common_data_checks(travel_times_mat, incidence_mat, obs_mat)

    def test_example_manual_loading_sparse_dok(self):
        """Dok is "prefered" and internal code should convert to csr when required"""
        travel_times_mat, incidence_mat, obs_mat = self.load_example_tiny_manually()
        travel_times_mat = travel_times_mat.todok()
        incidence_mat = incidence_mat.todok()
        obs_mat = obs_mat.todok()
        self._first_example_common_data_checks(travel_times_mat, incidence_mat, obs_mat)

    def test_example_manual_loading_dense(self):
        travel_times_mat, incidence_mat, obs_mat = self.load_example_tiny_manually()

        travel_times_mat = travel_times_mat.toarray()
        incidence_mat = incidence_mat.toarray()
        obs_mat = obs_mat.toarray()

        self._first_example_common_data_checks(travel_times_mat, incidence_mat, obs_mat)

    def test_example_tiny_smart_loading(self):
        subfolder = "ExampleTiny"  # big data from classical v2
        folder = join("Datasets", subfolder)
        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=False)
        incidence_mat, travel_times_mat = attrs
        # left, right, _, u_turn = AngleProcessor.get_turn_categorical_matrices()
        self._first_example_common_data_checks(travel_times_mat, incidence_mat, obs_mat)

    @staticmethod
    def _tiny_modified_common_data_checks(travel_times_mat,
                                          left, u_turn, t_time_incidence,
                                          incidence_mat, obs_record):
        data_list = [travel_times_mat, left, u_turn, t_time_incidence]
        network_data_struct = ModelDataStruct(data_list, incidence_mat)

        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = optimisers.LineSearchOptimiser(optimisers.OptimHessianType.BFGS, max_iter=4)
        RecursiveLogitModelEstimation.zeros_error_override = False
        # hack
        model = RecursiveLogitModelEstimation(network_data_struct, optimiser,
                                              observations_record=obs_record)
        log_like_out, grad_out = model.get_log_likelihood()
        eps = 1e-6

        print(optimiser.get_iteration_log(
            model.optim_function_state))  # Note this is currently required to
        # set the gradient so that compute relative gradient works, really bad
        # model.hessian = np.identity(network_data_struct.n_dims)
        # Pre  @42f564e9, we are manipulating behaviour slightly so this test still works
        # should just replace with Sioux falls or similar
        ll, line_search_step, grad_norm, rel_grad_norm = (2.079441541679836, 0.0,
                                                          0.7071067811865476,
                                                          0.24044917348149386)
        assert np.abs(log_like_out - ll) < eps
        assert np.abs(model.optimiser.step - line_search_step) < eps

        assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
        assert np.abs(model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

        targets = [
            (1.699556, 0.7071068, 0.3803406, 0.1582422),
            (1.495032, 0.8230394, 0.1457128, 0.06891792),
            (1.440549, 0.5111388, 0.07468365, 0.03665915)
        ]
        for t in targets:
            ll, line_search_step, grad_norm, rel_grad_norm = t
            out_flag, hessian, log = optimiser.iterate_step(model.optim_function_state,
                                                            verbose=False)
            log_like_out, grad_out = model.get_log_likelihood()

            print(f"({log_like_out:.7}, "
                  f"{linalg.norm(model.optimiser.step):.7}, {linalg.norm(grad_out):.7}, "
                  f"{model.optimiser.compute_relative_gradient_non_static():.7})")

            assert np.abs(log_like_out - ll) < eps
            assert np.abs(linalg.norm(model.optimiser.step) - line_search_step) < eps
            assert np.abs(linalg.norm(grad_out) - grad_norm) < eps
            assert np.abs(
                model.optimiser.compute_relative_gradient_non_static() - rel_grad_norm) < eps

        RecursiveLogitModelEstimation.zeros_error_override = None  # reset

    def test_example_tiny_modified_sparse(self):
        # TODO shouldn't be using this data - data is just confusing/ unclear, nothing inherently
        #  wrong
        # Now is a bad example as  @42f564e9 results in this test case being illegal valued
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times
        # - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        self._tiny_modified_common_data_checks(travel_times_mat,
                                               left, u_turn, t_time_incidence,
                                               incidence_mat, obs_mat)

    def test_example_tiny_modified_awkward_array(self):
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_record = ak.from_numpy(obs_mat)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times
        # - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        self._tiny_modified_common_data_checks(travel_times_mat,
                                               left, u_turn, t_time_incidence,
                                               incidence_mat, obs_record)

    def test_example_tiny_modified_awkward_array_in_expected_format(self):
        """Test's awkward array input obs format when it is actually zero indexed and ragged
        data, not square. See that output is consistent in this case"""
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_list_raw = obs_mat.tolist()
        # we know that obs mat is square and 1 indexed with zero for padding (sparse originally)
        obs_conv = [[(i - 1) for i in row if i != 0] for row in obs_list_raw]
        obs_record = ak.from_iter(obs_conv)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        # incidence matrix which only has nonzero travel times
        # - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int').todok()
        self._tiny_modified_common_data_checks(travel_times_mat,
                                               left, u_turn, t_time_incidence,
                                               incidence_mat, obs_record)

    def test_example_tiny_modified_dense(self):
        subfolder = "ExampleTinyModifiedObs"  # big data from classical v2
        folder = join("Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_record = ak.from_numpy(obs_mat)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs

        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)

        # incidence matrix which only has nonzero travel times
        # - rather than what is specified in file
        t_time_incidence = (travel_times_mat > 0).astype('int')
        self._tiny_modified_common_data_checks(travel_times_mat.toarray(),
                                               left.toarray(), u_turn.toarray(),
                                               t_time_incidence.toarray(),
                                               incidence_mat.toarray(), obs_record)


class DataTransformsTest(object):
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


class TestSimulation(object):

    @staticmethod
    def _get_basic_consistency_expected(allow_positive_value_funcs):
        if allow_positive_value_funcs:
            return [
                [1, 0, 4, 5, 1], [1, 0, 4, 1, 1], [1, 0, 1], [1, 0, 0, 4, 5, 1], [1, 2, 6, 1],
                [1, 2, 1], [1, 2, 1], [1, 2, 1], [1, 7, 5, 1], [1, 7, 6, 1], [1, 7, 7, 5, 6, 1],
                [1, 7, 3, 4, 1], [6, 0, 4, 6], [6, 0, 1, 1, 6], [6, 0, 4, 6], [6, 0, 4, 6],
                [6, 1, 4, 6], [6, 1, 6], [6, 1, 6], [6, 1, 5, 1, 6], [6, 2, 7, 6], [6, 2, 6],
                [6, 2, 7, 6], [6, 2, 6], [6, 7, 6], [6, 7, 6, 5, 4, 6], [6, 7, 5, 1, 6],
                [6, 7, 6], [3, 0, 3], [3, 0, 3], [3, 0, 3], [3, 0, 3], [3, 1, 5, 3],
                [3, 1, 4, 3], [3, 1, 5, 3], [3, 1, 4, 3], [3, 2, 7, 3], [3, 2, 7, 3],
                [3, 2, 7, 3], [3, 2, 7, 7, 3], [3, 7, 7, 3], [3, 7, 3], [3, 7, 3, 3, 3, 3],
                [3, 7, 3]]
        else:
            return [
                [1, 0, 4, 5, 1], [1, 0, 1], [1, 0, 4, 1], [1, 0, 1], [1, 2, 1], [1, 2, 1],
                [1, 2, 1], [1, 2, 1], [1, 7, 7, 7, 5, 1], [1, 7, 6, 1], [1, 7, 3, 4, 1],
                [1, 7, 5, 1], [6, 0, 1, 6], [6, 0, 1, 6], [6, 0, 1, 6], [6, 0, 3, 7, 3, 7, 6],
                [6, 1, 4, 6], [6, 1, 6], [6, 1, 6], [6, 1, 6], [6, 2, 6], [6, 2, 6], [6, 2, 6],
                [6, 2, 6], [6, 7, 5, 4, 6], [6, 7, 6], [6, 7, 5, 6], [6, 7, 6], [3, 0, 3],
                [3, 0, 3], [3, 0, 3], [3, 0, 3], [3, 1, 4, 3], [3, 1, 0, 3], [3, 1, 4, 3],
                [3, 1, 4, 3], [3, 2, 7, 3], [3, 2, 7, 7, 3], [3, 2, 7, 3], [3, 2, 7, 7, 3],
                [3, 7, 3], [3, 7, 3], [3, 7, 3], [3, 7, 3]]

    @staticmethod
    def _basic_consistencey_checks(distances):
        data_list = [distances]
        if issparse(distances):
            hand_net_incidence_local = dok_matrix(hand_net_incidence)
        else:
            hand_net_incidence_local = hand_net_incidence
        network_struct = ModelDataStruct(data_list, hand_net_incidence_local,
                                         data_array_names_debug=("distances", "u_turn"))
        beta_vec = np.array([-1])
        model = RecursiveLogitModelPrediction(network_struct,
                                              initial_beta=beta_vec, mu=1)
        obs = model.generate_observations(origin_indices=[0, 1, 2, 7], dest_indices=[1, 6, 3],
                                          num_obs_per_pair=4, iter_cap=15, rng_seed=1)
        expected = TestSimulation._get_basic_consistency_expected(ALLOW_POSITIVE_VALUE_FUNCTIONS)
        assert obs == expected

    def test_basic_consistency_sparse(self):

        distances = dok_matrix(hand_net_dists)
        self._basic_consistencey_checks(distances)

    def test_basic_consistency_dense(self):

        distances = hand_net_dists
        self._basic_consistencey_checks(distances)

    def test_invalid_beta_throws(self):
        distances = dok_matrix(hand_net_dists)
        if issparse(distances):
            hand_net_incidence_local = dok_matrix(hand_net_incidence)
        else:
            hand_net_incidence_local = hand_net_incidence

        data_list = [distances]
        network_struct = ModelDataStruct(data_list, hand_net_incidence_local,
                                         data_array_names_debug=("distances", "u_turn"))

        beta_vec = np.array([-5])

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
        folder = join("Datasets", subfolder)

        obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=True)
        import awkward1 as ak
        obs_mat = obs_mat.toarray()
        obs_record = ak.from_numpy(obs_mat)
        incidence_mat, travel_times_mat, angle_cts_mat = attrs
        left, _, _, u_turn = AngleProcessor.get_turn_categorical_matrices(angle_cts_mat,
                                                                          incidence_mat)
        data_list = [travel_times_mat, left, u_turn]
        network_data_struct = ModelDataStruct(data_list, incidence_mat)

        # network_data_struct.add_second_travel_time_for_testing()
        optimiser = optimisers.LineSearchOptimiser(optimisers.OptimHessianType.BFGS, max_iter=4)
        RecursiveLogitModelEstimation.zeros_error_override = False
        model = RecursiveLogitModelEstimation(network_data_struct, optimiser,
                                              observations_record=obs_record,
                                              initial_beta=-15)

        m1_ll_out, m1_grad_out = model.get_log_likelihood()

        optimiser2 = optimisers.ScipyOptimiser(method='newton-cg')

        model2 = RecursiveLogitModelEstimation(network_data_struct, optimiser2,
                                               observations_record=obs_record,
                                               initial_beta=-15)
        m2_ll_out, m2_grad_out = model2.get_log_likelihood()

        assert np.allclose(m2_ll_out, m1_ll_out)
        assert np.allclose(m2_grad_out, m1_grad_out)

        beta1 = model.solve_for_optimal_beta()

        beta2 = model2.solve_for_optimal_beta(verbose=True)
        m1_ll_out, m1_grad_out = model.get_log_likelihood()
        m2_ll_out, m2_grad_out = model2.get_log_likelihood()
        print(m1_ll_out, m2_ll_out)
        print(m1_grad_out, m2_grad_out)

        assert np.allclose(beta1, beta2, 0.34657)

        RecursiveLogitModelEstimation.zeros_error_override = None
