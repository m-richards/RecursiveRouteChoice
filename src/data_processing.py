"""file for processing matrices after they have been loaded from file,
or generate by hand/ other means.

Currently just for processing angles"""
import numpy as np
import scipy
from scipy.sparse import coo_matrix
PI = 3.1415926535
LEFT_TURN_THRESH = -30 * PI/180  # 30 degrees
U_TURN_THRESH = 177 * PI/180  # radians # TODO this is very tight
RIGHT_TURN_THRESH = -LEFT_TURN_THRESH

# NEAR_2PI_MULTIPLIER = 1.95
ALMOST_2_PI = 1.95 * PI


# TODO dense matrix optimised routines
# TODO no casting should be done here
class AngleProcessor(object):

    @classmethod
    def get_turn_categorical_matrices(cls, turn_angles_cts, incidence_mat=None,
                                      u_turn_thresh=None, left_turn_thresh=None):
        left = cls.get_left_turn_categorical_matrix(turn_angles_cts, left_turn_thresh,
                                                    u_turn_thresh)
        right_turn_thresh = -left_turn_thresh if left_turn_thresh is not None else None
        right = cls.get_right_turn_categorical_matrix(turn_angles_cts, right_turn_thresh,
                                                      u_turn_thresh)
        u_turn = cls.get_u_turn_categorical_matrix(turn_angles_cts, u_turn_thresh)
        neutral = cls.get_neutral_turn_categorical_matrix(turn_angles_cts, left_turn_thresh)
        return left, right, neutral, u_turn

    @classmethod
    def get_u_turn_categorical_matrix(cls, turn_angle_mat, u_turn_thresh=None):
        """Assumes that angles are between -pi and pi"""
        u_turn_thresh = u_turn_thresh if u_turn_thresh is not None else U_TURN_THRESH

        nz_rows, nz_cols = turn_angle_mat.nonzero()

        nz_u_turns_mask = np.array(
            (turn_angle_mat[nz_rows, nz_cols].toarray() > u_turn_thresh) &  # turn is to the
            # left
            (turn_angle_mat[nz_rows, nz_cols].toarray() < ALMOST_2_PI))[0]  # turn is not a
        # uturn
        # note testing todense suggests faster or at least not worse, supresses error
        masked_rows = nz_rows[nz_u_turns_mask]
        masked_cols = nz_cols[nz_u_turns_mask]
        vals = np.ones(len(masked_cols), dtype='int')
        u_turn_mat = coo_matrix(
            (vals, (masked_rows, masked_cols)), shape=turn_angle_mat.shape, dtype='int')

        return u_turn_mat.todok()

    @classmethod
    def get_left_turn_categorical_matrix(cls, turn_angle_mat, left_turn_thresh=None,
                                         u_turn_thresh=None):
        """Assumes that angles are between -pi and pi"""
        if left_turn_thresh is None:
            left_turn_thresh = LEFT_TURN_THRESH
        if u_turn_thresh is None:
            u_turn_thresh = U_TURN_THRESH
        # Note this is done strangely since scipy doesn't support & conditions on
        # sparse matrices. Also is more efficient to only
        # do comparison on nonzero (since this is dense)
        nz_rows, nz_cols = turn_angle_mat.nonzero()

        nz_left_turns_mask = np.array(
            (turn_angle_mat[nz_rows, nz_cols].toarray() < left_turn_thresh) &  # turn is to the left
            (turn_angle_mat[nz_rows, nz_cols].toarray() > -u_turn_thresh))[0]  # turn is not a uturn
        masked_rows = nz_rows[nz_left_turns_mask]
        masked_cols = nz_cols[nz_left_turns_mask]
        vals = np.ones(len(masked_cols), dtype='int')
        left_turn_mat = scipy.sparse.coo_matrix(
            (vals, (masked_rows, masked_cols)), shape=turn_angle_mat.shape, dtype='int')

        return left_turn_mat.todok()

    @classmethod
    def get_right_turn_categorical_matrix(cls, turn_angle_mat, right_turn_thresh=None,
                                          u_turn_thresh=None):
        """Assumes that angles are between -pi and pi"""
        if right_turn_thresh is None:
            right_turn_thresh = RIGHT_TURN_THRESH
        if u_turn_thresh is None:
            u_turn_thresh = U_TURN_THRESH
        # Note this is done strangely since scipy doesn't support & conditions on
        # sparse matrices. Also is more efficient to only do comparison on nonzero
        # (since this is dense)
        nz_rows, nz_cols = turn_angle_mat.nonzero()

        nz_right_turns_mask = np.array(
            (turn_angle_mat[nz_rows, nz_cols].toarray() > right_turn_thresh) &  # turn is to the
            # right
            (turn_angle_mat[nz_rows, nz_cols].toarray() < u_turn_thresh))[0]  # turn is not a uturn
        masked_rows = nz_rows[nz_right_turns_mask]
        masked_cols = nz_cols[nz_right_turns_mask]
        vals = np.ones(len(masked_cols), dtype='int')
        right_turn_mat = scipy.sparse.coo_matrix(
            (vals, (masked_rows, masked_cols)), shape=turn_angle_mat.shape, dtype='int')

        return right_turn_mat.todok()

    @classmethod
    def get_neutral_turn_categorical_matrix(cls, turn_angle_mat,
                                            side_turn_thresh=None):
        """
        We assume that all zero angles are absent arcs, genuine zero angles should be encoded
        as near zero nonzero - i.e. 0.01 or as 2* Pi =  6.28 = 0
        # IF this is not the case it should be handled otherwise first

        :param turn_angle_mat:
        :type turn_angle_mat:
        :param side_turn_thresh: left or right turn threshold, assumed to be symmetric
        :type side_turn_thresh:
        :param u_turn_thresh: u turn threshold angle in radians
        :type u_turn_thresh:
        """
        NON_ZERO_NEAR_ZERO = 0.01
        if side_turn_thresh is None:
            side_turn_thresh = LEFT_TURN_THRESH
        side_turn_thresh = abs(side_turn_thresh)
        # print(turn_angle_mat.toarray())

        # rezero any entries which are encoded as 2 * Pi
        turn_angle_mat[
            turn_angle_mat > ALMOST_2_PI] = NON_ZERO_NEAR_ZERO  # rezero these
        # print("modif")
        # print(turn_angle_mat[turn_angle_mat > 0.95 * PI].toarray())

        # genuine_zeros = turn_angle_mat[incidence_mat]

        # print(turn_angle_mat.toarray())
        # Note this is done strangely since scipy doesn't support & conditions on
        # sparse matrices. Also is more efficient to only do comparison on
        # nonzero (since this is dense)
        nz_rows, nz_cols = turn_angle_mat.nonzero()
        nz_left_turns_mask = np.array(
            # not a left turn
            (turn_angle_mat[nz_rows, nz_cols].toarray() > -side_turn_thresh) &
            # not a right turn   # turn is not a uturn
            (turn_angle_mat[nz_rows, nz_cols].toarray() < side_turn_thresh))[0]
        # note testing todense suggests faster or at least not worse, supresses error
        masked_rows = nz_rows[nz_left_turns_mask]
        masked_cols = nz_cols[nz_left_turns_mask]
        vals = np.ones(len(masked_cols), dtype='int')
        neutral_turn_mat = coo_matrix(
            (vals, (masked_rows, masked_cols)), shape=turn_angle_mat.shape, dtype='int')
        # return element wise product with incident mat
        return neutral_turn_mat.todok()

    @classmethod
    def to_radians(cls, angle_turn_mat):

        return angle_turn_mat * PI / 180.0


def get_incorrect_tien_turn_matrices(turn_angle_mat, left_turn_thresh=LEFT_TURN_THRESH,
                                     u_turn_thresh=U_TURN_THRESH):
    """A function to generate the turn matrices equivalently tien mai code for comparison
    purposes, note that this is logically incorrect though.
    Deliberately copies confusing logic since it is trying to be consistent.
    Computes turn angles correctly in a convoluted way.
    Skips computing leftTurn since this is overridden to be an incidence matrix"""
    # Angles between -pi and pi
    u_turn_mat = (np.abs(turn_angle_mat) > u_turn_thresh).astype(int)

    new_turn_angles = turn_angle_mat.copy()
    nonzero_turn_angles = np.nonzero(turn_angle_mat)
    for x, y in zip(*nonzero_turn_angles):
        i = (x, y)
        current_turn_angle = turn_angle_mat[i]
        if abs(current_turn_angle) < u_turn_thresh:  # not a uturn
            if current_turn_angle >= 0:  # turn to the right
                new_turn_angles[i] = 0  # remove right turns from matrix (not sure why)
            else:  # straight or to the left
                if current_turn_angle < left_turn_thresh:
                    new_turn_angles[i] = 1
                else:
                    new_turn_angles[i] = 0
        else:
            new_turn_angles[i] = 0

    return new_turn_angles, u_turn_mat
