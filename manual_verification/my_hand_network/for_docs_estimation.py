import optimisers
from recursive_route_choice import RecursiveLogitModelEstimation

obs = []  # from other file
network_struct = []  # from other file

optimiser = optimisers.ScipyOptimiser(method='bfgs')

model = RecursiveLogitModelEstimation(network_struct, observations_record=obs,
                                      initial_beta=[-5], mu=1,
                                      optimiser=optimiser)
beta = model.solve_for_optimal_beta(verbose=False)

print("Optimal beta determined was", beta)
