
import nibabel as nib
import numpy as np
# import matplotlib as plt
# import pandas as pd
import math as math
# import os,time,cv2, sys
import os
from PIL import Image
from tifffile import imsave
import scipy.io as scio
import glob as glob
from skimage.transform import resize

if __name__ == '__main__':
    # takes the niftii, saves the tif's, saves the 3D images of each channel at all time points
    time = 1;
    z = 15;
    originalImageName = 'EcadMyo_08'
    originalImageAddress = 'C:/Users/Gautam/Desktop/oneone/'
    originalImage = nib.load(originalImageAddress + originalImageName + '.nii')

    originalImageFloat32 = np.asarray(originalImage.dataobj).astype(np.float32).squeeze()

    originalImageSize = np.shape(originalImage);
    protein1name = 'Ecad';
    protein2name = 'Myosin';

    if not os.path.isdir(originalImageAddress+"3DImage"):
        os.makedirs(originalImageAddress+"3DImage")

    dirp1 = originalImageAddress + '3DImage/' + originalImageName + '/' + protein1name
    dirp2 = originalImageAddress + '3DImage/' + originalImageName + '/' + protein2name

    if not os.path.isdir(dirp1):
        os.makedirs(dirp1)
    if not os.path.isdir(dirp2):
        os.makedirs(dirp2)

    print(originalImageSize)

    for i in range (0,originalImageSize[3]):
        if(i<9):
            ttag = '000'
        elif (i<99):
            ttag = '00'
        else:
            ttag = '0'

        slice1 = originalImage.slicer[:, :, :, i, 0]
        slice2 = originalImage.slicer[:, :, :, i, 1]

        nib.save(slice1, dirp1 + '/threeD_img' + ttag + str(i + 1))
        nib.save(slice2, dirp2 + '/threeD_img' + ttag + str(i + 1))

        for j in range (0,originalImageSize[2]):
            if(j<9):
                ztag = '000'
            elif (j<99):
                ztag = '00'
            else:
                ztag = '0'

            tifname1 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'
            tifname2 = dirp1 + '/' + originalImageName + '_t' + ttag + str(i + 1) + '_z' + ztag + str(j + 1) + '.tif'

            Xxxx = Image.fromarray(originalImageFloat32[:,:,j,i,0],mode='F')
            sliceA = Image.fromarray(np.asarray(Xxxx).squeeze())
            sliceA.save(tifname1)
            Xxxx = Image.fromarray(originalImageFloat32[:,:,j,i,0],mode='F')
            sliceB = Image.fromarray(np.asarray(Xxxx).squeeze())
            sliceB.save(tifname2)

            del sliceB, slice2, slice1, sliceA, originalImage,originalImageFloat32 #clear; free up some memory

    ####################################################################################################################

    colormap = scio.loadmat('C:/my3D_matlab/colormap.mat')
    t1 = 1;
    t2 = 41;

    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [35, 35, I3dw(2)]

    for time in range(t1-1,t2):
        print(time)
        tt = str(time)
        addr = 'D:/NEW\Prediction_Result_Ajuba_09/Prediction_Dataset_Ajuba_sqh-cherry_jub-gfp 18A-09/FC-DenseNet/' + str(time) + '/'
        addr2 = 'D:/NEW/Recombined_18A-09/'+ str(time) + '/'

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = glob.glob(addr + '*.nii')
        print(Files1)
        Fullsize = np.zeros(512, 280, 15)
        Fullsize_regression = np.zeros(512, 280, 15)
        Fullsize_input = np.zeros(512, 280, 15)
        Weights = np.zeros(512, 280, 15, 64)

        c_file = 0

        for i1 in range(0,I3dw(0)-I3d(0),I3d(0)):
            for i2 in range(0,I3dw(1)-I3d(1),I3d(1)):
                V = nib.load(Files1(c_file))
                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
                V_arr = 1-V_arr
                V2 = np.uint8(V_arr*255)
                # v3 = imresize3(V2,I3d,'linear')
                V3 = resize(V2,I3d,order=1)

# for i1=1:I3d(1): I3dw(1) - I3d(1) + 1
# for i2=1:I3d(2): I3dw(2) - I3d(2) + 1
# V = niftiread(strcat(addr, Files1(c_file).name));
# V = 1 - V; % Because
# the
#
#
# class dictionary was defined that way (background =1 and foreground =0)
#
#
# V2 = uint8(V * 255);
# V3 = imresize3(V2, I3d, 'linear');
#
# a = i1;
# b = i1 + I3d(1) - 1;
# c = i2;
# d = i2 + I3d(2) - 1;
# Fullsize(a: b, c: d,:)=V3;
#
# V = niftiread(strcat(addr, Files1(c_file + 1).name));
# for iy=1:64 % % all = 256
# V2 = double(V(:,:,:, iy));
# V3 = imresize3(V2, I3d, 'linear');
# Weights(a: b, c: d,:, iy)=V2;
# end
#
# V = niftiread(strcat(addr, Files1(c_file + 2).name));
# V3 = imresize3(V, I3d, 'linear');
# Fullsize_input(a: b, c: d,:)=V3;
# c_file = c_file + 4;
# end
# end