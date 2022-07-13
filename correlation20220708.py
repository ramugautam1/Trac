import numpy as np


def size3(arg):
    return [np.size(arg, axis=0), np.size(arg, axis=1), np.size(arg, axis=2)]


def correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2,
                t2, spatial_extend_matrix, addr2, padding, time=1):

    depth = np.size(Fullsize_regression_1, 3)

    # get the size of sample
    [x, y, z] = size3(Fullsize_1)
    [x_reserve, y_reserve, z_reserve] = size3(Fullsize_1)
    print([x, y, z])
    





