import scipy
import numpy as np

# scipy.seterr(all='raise')#all='print')

# working out how to do an and condition without and conditions
# want to get entries between 0 and -5
a = np.array([[1, 2, 0], [-2, -3, -4], [-7, -0, -9]])
print(a)
from scipy.sparse import csr_matrix
import warnings

# warnings.simplefilter("error")
b = csr_matrix(a)
print(b)

f = scipy.sparse.find(b > 0)
print(":mask1")
print((b < 0).astype('int'))
mask1 = (0 > b).astype('int')
mask2 = (-5 < b)  # -5<b <==>

mask2_alt = b.nonzero()  # (rows, cols) which are nonzero
one_dim_nonzero = b[b.nonzero()]  # gives a 1D matrix of the nonzero entries
boolean_mask = np.array(b[b.nonzero()] < -3)[0]  # one dim boolean mask (convert from 1*n matrix to
# 1d array
rows = b.nonzero()[0][boolean_mask]
cols = b.nonzero()[1][boolean_mask]

joint_mask = mask1 + mask2
print("types, ", type(mask1), type(mask2), type(joint_mask))
# joint_mask = scipy.sparse.find(joint_mask==2)
final_mask = joint_mask == 2
print(final_mask, "\nz", type(final_mask))
print(joint_mask)
print(joint_mask.toarray())
print("test")
print(b[final_mask])
print(type(b[final_mask]))

print("indices mask", type(final_mask))
print(final_mask.toarray())

# BETTTER approach with no warnings
# mask_left_turn_or_left_uturn = (left_turn_thresh >= turn_angle_mat).astype('int')
# modify c to create boolean array, rather than using comparison operators
# c = scipy.sparse.identity(b.shape[0]).tolil(copy=False)
# c = b.copy()
nz_rows, nz_cols = b.nonzero()
mask = np.array(b[nz_rows, nz_cols] < -3)[0]

mask2 = np.array(b[nz_rows, nz_cols] > -7)[0]
combined = mask & mask2
masked_rows = nz_rows[combined]
masked_cols = nz_cols[combined]
# c[masked_rows, masked_cols] =1

# entries

d = scipy.sparse.coo_matrix((np.ones(len(masked_cols)), (masked_rows, masked_cols)),
                            shape=b.shape, dtype='int')

print(d.toarray())
