r"""
We consider the very simple network and verify behaviour makes sense and is consistent for
all input types. Network is as follows

/--\  <------4-----    /--\   <-----3------  /--\
|   |                 |   |                 |   |
\__/  -------1----->  \__/    ------2-----> \__/
with since attribute distance, Arc1: 4, Arc2:6, Arc3: 6, Arc4: 4.
We consider an arc formulation, so the distances become the average of these:
distances = np.array(
        1  2  3  4
-----------------------
  #1  [[0, 5, 0, 4],
  #2   [0, 0, 3, 0],   # 3's should be 6's
  #3   [0, 3, 0, 5],
  #4   [4, 0, 0, 0]])

3's instead of 6's was just an error in encoding, probably should be
fixed. However this is sufficient as a test case - it just doesn't
quite describe the network above (if you like there is some kind of
"short cut travelling from 2 to 3 and 3 to 2 (they don't
have to travel the whole way across an intersection?)
but this isn't important)
"""
import numpy as np
import pytest
from scipy.sparse import dok_matrix
import awkward1 as ak

from recursiveRouteChoice import ModelDataStruct, RecursiveLogitModelEstimation, \
    RecursiveLogitModelPrediction
from recursiveRouteChoice import optimisers

np.set_printoptions(edgeitems=10, linewidth=300)
# np.core.arrayprint._line_width = 500

# DATA
# silly deterministic network
distances = np.array(
    [[0, 5, 0, 4],
     [0, 0, 3, 0],  # 3's should be 6's mathematically
     [0, 3, 0, 5],
     [4, 0, 0, 0]])

distances = dok_matrix(distances)
incidence_mat = (distances > 0).astype(int)
network_struct = ModelDataStruct([distances], incidence_mat,
                                 data_array_names_debug=("distances",))
optimiser = optimisers.ScipyOptimiser(method='bfgs')

# this did come from generate obs with beta = -0.4, but that code might change with fixes
# obs = model.generate_observations(origin_indices=[0],
#                                   dest_indices=[1],
#                                   num_obs_per_pair=3, iter_cap=2000, rng_seed=1,
#                                   )
input_obs = [[1, 0, 1, 2, 1], [1, 0, 3, 0, 1], [1, 0, 1]]

model = RecursiveLogitModelEstimation(network_struct, observations_record=input_obs,
                                      initial_beta=[-0.4], mu=1,
                                      optimiser=optimiser)


class TestCases(object):

    def test_singleton_fails(self):
        with pytest.raises(TypeError) as excinfo:
            RecursiveLogitModelEstimation(network_struct, observations_record=input_obs[0],
                                          initial_beta=-0.4, mu=1,
                                          optimiser=optimiser)
        assert "List observation format must contain list of lists" in str(excinfo.value)

    def test_obs1(self):
        model = RecursiveLogitModelEstimation(network_struct, observations_record=[input_obs[0]],
                                              initial_beta=-0.4, mu=1,
                                              optimiser=optimiser)
        ll = model.get_log_likelihood()[0]
        start_link_ind, fin_link_ind = model._compute_obs_path_indices(ak.from_iter(input_obs[0]))
        assert tuple(start_link_ind) == (0, 1, 2)
        assert tuple(fin_link_ind) == (1, 2, 1)
        assert (pytest.approx(model.get_short_term_utility()[start_link_ind, fin_link_ind].sum()) ==
                -4.4)
        assert pytest.approx(ll, abs=1e-6) == 2.537993985

    def test_obs2(self):
        obs = [input_obs[1]]
        model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=-0.4, mu=1,
                                              optimiser=optimiser)
        ll = model.get_log_likelihood()[0]

        start_link_ind, fin_link_ind = model._compute_obs_path_indices(ak.from_iter(obs[0]))
        assert tuple(start_link_ind) == (0, 3, 0)
        assert tuple(fin_link_ind) == (3, 0, 1)
        assert (pytest.approx(model.get_short_term_utility()[start_link_ind, fin_link_ind].sum())
                == -5.2)

        print(model.get_short_term_utility()[start_link_ind, fin_link_ind].sum())
        assert pytest.approx(ll) == 3.337993985

    def test_obs3(self):
        obs = [input_obs[2]]
        model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                              initial_beta=-0.4, mu=1,
                                              optimiser=optimiser)
        ll = model.get_log_likelihood()[0]

        start_link_ind, fin_link_ind = model._compute_obs_path_indices(ak.from_iter(obs[0]))
        assert tuple(start_link_ind) == (0,)
        assert tuple(fin_link_ind) == (1,)
        assert (pytest.approx(model.get_short_term_utility()[start_link_ind, fin_link_ind].sum())
                == -2)
        assert pytest.approx(ll) == 0.137993985

    def test_obs_all(self):
        model = RecursiveLogitModelEstimation(network_struct, observations_record=input_obs,
                                              initial_beta=-0.4, mu=1,
                                              optimiser=optimiser)
        ll = model.get_log_likelihood()[0]
        assert pytest.approx(ll) == 6.01398195541


# bigger silly network - see phone photo
@pytest.fixture
def struct_bigger():
    distances = np.array(
        [[0, 5, 0, 4, 0, 0, 0, 0, 0, 0],
         [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
         [0, 6, 0, 5, 0, 0, 0, 0, 0, 0],
         [4, 0, 0, 0, 5, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
         [5, 0, 0, 0, 6, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 6, 6, 0],
         [0, 0, 0, 0, 0, 6, 6, 0, 0, 0],
         [0, 0, 6, 0, 0, 0, 0, 0, 0, 6],
         [0, 0, 0, 0, 0, 0, 0, 6, 6, 0]
         ])
    distances = dok_matrix(distances)
    incidence_mat = (distances > 0).astype(int)
    network_struct = ModelDataStruct([distances], incidence_mat,
                                     data_array_names_debug=("distances",))
    return network_struct


class TestPredictionExceptions(object):

    def test_bad_beta_fails(self, struct_bigger):

        model = RecursiveLogitModelPrediction(struct_bigger,
                                              initial_beta=-0.1)
        with pytest.raises(ValueError) as e:
            model.generate_observations(origin_indices=[0], dest_indices=[9], num_obs_per_pair=10)
        assert "exp(V(s)) contains negative values" in str(e.value)

    def test_bad_indexfails(self, struct_bigger):
        model = RecursiveLogitModelPrediction(struct_bigger,
                                              initial_beta=-0.2)
        with pytest.raises(IndexError) as e:
            model.generate_observations(origin_indices=[0], dest_indices=[100], num_obs_per_pair=10)
        assert "Can only simulate observations from indexes which are in the model" in str(e.value)

        with pytest.raises(IndexError) as e:
            model.generate_observations(origin_indices=[0], dest_indices=[10], num_obs_per_pair=10)
        assert "but the final index is reserved for internal dummy sink state" in str(e.value)
