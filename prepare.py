# import cv2
import glob as glob
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


def prepare():
    # takes the niftii, saves the tif's, saves the 3D images of each channel at all time points
    time = 1;
    z = 15;
    originalImageName = 'EcadMyo_08'
    originalImageAddress = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/'
    originalImage = nib.load(originalImageAddress + originalImageName + '.nii')

    originalImageFloat32 = np.asarray(originalImage.dataobj).astype(np.float32).squeeze()
    print(np.shape(originalImageFloat32))

    originalImageSize = np.shape(originalImage);
    protein1name = 'Ecad';
    protein2name = 'Myosin';

    if not os.path.isdir(originalImageAddress + "3DImage"):
        os.makedirs(originalImageAddress + "3DImage")

    dirp1 = originalImageAddress + '3DImage/' + originalImageName + '/' + protein1name
    dirp2 = originalImageAddress + '3DImage/' + originalImageName + '/' + protein2name

    if not os.path.isdir(dirp1):
        os.makedirs(dirp1)
    if not os.path.isdir(dirp2):
        os.makedirs(dirp2)

    print(originalImageSize)

    for i in range(0, originalImageSize[3]):
        if (i < 9):
            ttag = '000'
        elif (i < 99):
            ttag = '00'
        else:
            ttag = '0'

        slice1 = originalImage.slicer[:, :, :, i, 0]
        slice2 = originalImage.slicer[:, :, :, i, 1]

        nib.save(slice1, dirp1 + '/threeD_img' + ttag + str(i + 1))
        nib.save(slice2, dirp2 + '/threeD_img' + ttag + str(i + 1))

        for j in range(0, originalImageSize[2]):
            if (j < 9):
                ztag = '000'
            elif (j < 99):
                ztag = '00'
            else:
                ztag = '0'

            tifname1 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'
            tifname2 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'

            Xxxx = Image.fromarray(originalImageFloat32[:, :, j, i, 0], mode='F')
            sliceA = Image.fromarray(np.asarray(Xxxx).squeeze())
            sliceA.save(tifname1)
            Xxxx = Image.fromarray(originalImageFloat32[:, :, j, i, 0], mode='F')
            sliceB = Image.fromarray(np.asarray(Xxxx).squeeze())
            sliceB.save(tifname2)

    del sliceB, slice2, slice1, sliceA, originalImage, originalImageFloat32  # clear; free up some memory
    ####################################################################################################################
