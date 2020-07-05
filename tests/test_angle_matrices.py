# Currently ignoring the problem of how these are generated in a correct fashion, we just want to
# say that the angles matrices are right if these are the given angles between a and b
import numpy as np
import scipy
from scipy.sparse import csr_matrix
U_TURN_THRESH = 3.1  # radians
def local_get_uturn_categorical_matrix(turn_angle_mat, u_turn_thresh=U_TURN_THRESH):
    """Assumes that angles are between -pi and pi"""
    return (np.abs(turn_angle_mat) > u_turn_thresh).astype(int)


a= np.array([[0 , -0.1, 180,],
             [90, 0, -90],
             [-45, -15, 0]])

b = a * np.pi/180
b = csr_matrix(b)
from data_loading import get_left_turn_categorical_matrix, get_uturn_categorical_matrix

left_turn = get_left_turn_categorical_matrix(b)
print("left turn")
print(left_turn.toarray())
print("uturn")

print(get_uturn_categorical_matrix(b).toarray())