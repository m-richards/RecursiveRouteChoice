import time
import numpy as np

from data_loading import load_standard_path_format_csv
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStruct

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
obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=False)
incidence_mat, travel_times_mat = attrs
data_list =[travel_times_mat, travel_times_mat]
network_data_struct = RecursiveLogitDataStruct(data_list, incidence_mat)


optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS, max_iter=4)

model = RecursiveLogitModelEstimation(network_data_struct, optimiser, user_obs_mat=obs_mat)

model.solve_for_optimal_beta()

time_finish = time.time()
print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}")
