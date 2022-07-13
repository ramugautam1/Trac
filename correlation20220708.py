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

    #padding the sample for 'extended search' (Fullsize: object label map, Fullsize_regression: object deep feature map)
    #Create the zeros the size of original plus padding
    
    Fullsize_1_padding = np.zeros(x+padding[0]*2, y+ padding[1]*2, z+padding[2]*2)
    Fullsize_2_padding = np.zeros(x+padding[0]*2, y+ padding[1]*2, z+padding[2]*2)
    
    Fullsize_regression_1_padding = np.zeros(x+padding[0]*2, y+padding[1]*2, z+padding[2]*2, depth)
    Fullsize_regression_2_padding = np.zeros(x+padding[0]*2, y+padding[1]*2, z+padding[2]*2, depth)
    
    #copy the originals to the middle, effectively getting zero padding around them
    #
    #
    #
    #
    
    correlation_map_padding_corr = np.zeros(x+padding[0]*2, y+padding[1]*2, z+padding[2]*2)
    correlation_map_padding_show = np.zeros(x+padding[0]*2, y+padding[1]*2, z+padding[2]*2)
    
    del Fullsize_regression_1, Fullsize_regression_2, Fullsize_1, Fullsize_2
    
    
    





