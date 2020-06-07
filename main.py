"""
Sketch of what we need

Class to store data
    travel time
    turn stuff
    (link size)

    keep track of optimisation progress - number of function evaluations

    - function to compute likelihood estimates
    - keep track of hyper parameters
     print some feedback on each operation - PrintOut function
        log likelihood
        values of beta
        norm of step
        radius
        norm & relative norm of grad
        number of evaluations

    log results on each iteration

    - optimisation algorithm LINE SEARCH for now?
    - stopping condition

    -


"""
import numpy as np

class DataSet:
    """Really just a struct"""

    def __init__(self, travel_times, incidence_matrix, turn_angles=None):
        self.travel_times = travel_times
        self.incidence_matrix = incidence_matrix
        self.turn_angles = turn_angles
        # TODO include uturn and left turn penalties when relevant
        self.attrs = [self.travel_times,]
        self.n_dims = len(self.attrs)
