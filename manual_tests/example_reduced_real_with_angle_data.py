import time
import numpy as np
from main import RecursiveLogitModelEstimation, RecursiveLogitDataStructDeprecated

import os
import optimisers as op
from optimisers import OptimType

np.seterr(all='raise')  # all='print')
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(edgeitems=3)
np.core.arrayprint._line_width = 100
import warnings

warnings.simplefilter("error")

time_io_start = time.time()
# file ="ExampleTutorial"# "ExampleTutorial" from classical logicv2
# file = "ExampleTiny"  # "ExampleNested" from classical logit v2, even smaller network
subfolder = "TienMaiRealDataCut"  # big data from classical v2
folder = os.path.join("../Datasets", subfolder)


# TODO needs work to work - don't have current code for comparison angle types
# obs_mat, attrs = load_standard_path_format_csv(folder, delim=" ", angles_included=False)
# incidence_mat, travel_times_mat = attrs
# # left, right, _, u_turn = AngleProcessor.get_turn_categorical_matrices()
# data_list =[travel_times_mat, travel_times_mat]
# network_data_struct = RecursiveLogitDataStruct2(data_list, incidence_mat)


network_data_struct, obs_mat = RecursiveLogitDataStructDeprecated.from_directory(folder, add_angles=True,
                                                                                 angle_type='comparison',
                                                                                 delim='\t')

time_io_end = time.time()

optimiser = op.LineSearchOptimiser(op.OptimHessianType.BFGS,

                                   max_iter=4)
model = RecursiveLogitModelEstimation(network_data_struct, optimiser, observations_record=obs_mat)

log_like_out, grad_out = model.get_log_likelihood(n_obs_override=1)


n = 0
print("Initial Values:")
optimiser.set_beta_vec(model.beta_vec)
optimiser.set_current_value(log_like_out)
print(optimiser.get_iteration_log(model))
while n <= 1000:
    # optimiser.iter_count += 1
    if model.optimiser.METHOD_FLAG == OptimType.LINE_SEARCH:
        ok_flag, hessian, log_msg = optimiser.iterate_step(model, verbose=False)
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
    break
if n == 1000:
    print("Infinite loop happened somehow, shouldn't have happened")

time_finish = time.time()

print(f"IO time - {round(time_io_end - time_io_start, 3)}s")
print(f"Algorithm time - {round(time_finish - time_io_end, 3)}")
