import glob as glob
import cv2
import math as math
import os
import sys
import time
import matplotlib as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage.transform import resize
from tifffile import imsave

def trackStep1():
# if __name__ == '__main__':
    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    t1 = 1;
    t2 = 41;

    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [35, 35, I3dw[2]]
    print(I3d[2])
    # for time in range(t1 - 1, t2):
    for time in range(t1, t2 + 1):
        print(time)
        tt = str(time)
        addr = '/home/nirvan/Desktop/Projects/allFiles/Prediction_Dataset_Ajuba_sqh-cherry_jub-gfp 18A-09/FC-DenseNet/' + \
               str(time) + '/'
        print(addr)
        addr2 = '/home/nirvan/Desktop/Projects/allFiles/' + str(time) + '/'
        print(addr2)

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = glob.glob(addr + '*.nii')
        print(Files1)
        Fullsize = np.zeros(shape=(512, 280, 15))
        Fullsize_regression = np.zeros(shape=(512, 280, 15))
        Fullsize_input = np.zeros(shape=(512, 280, 15))
        Weights = np.zeros(shape=(512, 280, 15, 64))

        c_file = 0

        for i1 in range(0, I3dw[0] - I3d[0], I3d[0]):
            for i2 in range(0, I3dw[1] - I3d[1], I3d[1]):
                # V = nib.load(Files1[c_file])
                # V = 1 - V
                # V2 = np.uint8(V * 255)
                # V3 = resize(V2, I3d, order=1)
                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
                V_arr = 1 - V_arr
                V2_arr = np.uint8(V_arr * 255)
                V3_arr = resize(V2_arr, I3d, order=1)

                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]
                # Fullsize[a:b, c:d, :] = V3
                Fullsize[a:b, c:d, :] = V3_arr

                # V = nib.load(Files1[c_file + 1])
                V_arr = np.asarray(nib.load(Files1[c_file+1]).dataobj).astype(np.float32).squeeze()
                print(f'V_arr shape {np.shape(V_arr)}')
                for iy in range(0, 64):
                    # V2 = np.double(V[:, :, :, iy]);
                    # V3 = resize(V2, I3d, order=1)
                    # Weights[a:b, c:d, :, iy] = V2
                    V2_arr = np.double(V_arr[:, :, :, iy]);
                    V3_arr = resize(V2_arr, I3d, order=1)
                    Weights[a:b, c:d, :, iy] = V2_arr

                # V = nib.load(Files1(c_file + 2))
                # V3 = resize(V, I3d, order=1)
                # Fullsize_input[a:b, c:d, :] = V3
                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()
                V3_arr = resize(V_arr, I3d, order=1)
                Fullsize_input[a:b, c:d, :] = V3_arr
                c_file = c_file + 4

        print(Weights[1:10, 1:10, 1, 1])
