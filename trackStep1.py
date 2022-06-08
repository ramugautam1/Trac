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
    colormap = scio.loadmat('C:/my3D_matlab/colormap.mat')
    t1 = 1;
    t2 = 41;

    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [35, 35, I3dw(2)]

    for time in range(t1 - 1, t2):
        print(time)
        tt = str(time)
        addr = 'D:/NEW\Prediction_Result_Ajuba_09/Prediction_Dataset_Ajuba_sqh-cherry_jub-gfp 18A-09/FC-DenseNet/' + \
               str(time) + '/'
        addr2 = 'D:/NEW/Recombined_18A-09/' + str(time) + '/'

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = glob.glob(addr + '*.nii')
        print(Files1)
        Fullsize = np.zeros(512, 280, 15)
        Fullsize_regression = np.zeros(512, 280, 15)
        Fullsize_input = np.zeros(512, 280, 15)
        Weights = np.zeros(512, 280, 15, 64)

        c_file = 0

        for i1 in range(0, I3dw(0) - I3d(0), I3d(0)):
            for i2 in range(0, I3dw(1) - I3d(1), I3d(1)):
                V = nib.load(Files1(c_file))
                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
                V_arr = 1 - V_arr
                V2 = np.uint8(V_arr * 255)
                # v3 = imresize3(V2,I3d,'linear')
                V3 = resize(V2, I3d, order=1)