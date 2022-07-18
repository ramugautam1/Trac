import glob as glob
import cv2
import math as math
import os
import sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage.transform import resize
from skimage import measure
from skimage import morphology
import matplotlib as mpl
import cc3d
from mpl_toolkits.mplot3d import Axes3D

from functions import niftiwrite, dashline


def trackStep1():
    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    t1 = 1
    t2 = 41;
    # t2 = 1
    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [32, 35, I3dw[2]]
    # for time in range(t1 - 1, t2):
    for time in range(t1, t2 + 1):
        tt = str(time)
        addr = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/Segmentation_Result_EcadMyo_08/EcadMyo_08/FC-DenseNet/'+ tt + '/'
        print(addr)
        addr2 = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08_Tracking_Result/' + str(time) + '/'
        print(addr2)

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = sorted(glob.glob(addr + '*.nii'))
        # print(Files1)
        Fullsize = np.zeros(shape=(512, 280, 15))
        Fullsize_regression = np.zeros(shape=(512, 280, 15))
        Fullsize_input = np.zeros(shape=(512, 280, 15))
        Weights = np.zeros(shape=(512, 280, 15, 64))

        c_file = 0

        for i1 in range(0, I3dw[0], I3d[0]):
            for i2 in range(0, I3dw[1], I3d[1]):

                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
                V_arr = 1 - V_arr
                V2_arr = np.uint8(V_arr * 255)
                V3_arr = resize(V2_arr, I3d, order=0)

                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]

                # print(f'{a}:{b}, {c}:{d}, : ')

                Fullsize[a:b, c:d, :] = V3_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()

                for iy in range(0, 64):
                    V2_arr = np.double(V_arr[:, :, :, iy])
                    V3_arr = resize(V2_arr, I3d, order=0)
                    Weights[a:b, c:d, :, iy] = V2_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 2]).dataobj)#.astype(np.float32).squeeze()
                V3_arr = resize(V_arr, I3d, order=0)
                Fullsize_input[a:b, c:d, :] = V3_arr.squeeze()
                c_file = c_file + 4

        #Remove small itty bitty masks
        Fullsize2 = Fullsize.astype(bool)
        # for it in range(0, np.size(Fullsize,2)):
        #     img = Fullsize2[:,:,it]
        #     f,orgnum = measure.label(img, connectivity=2, return_num=True)
        #     # g = measure.regionprops(f, 'area')

        # plt.imshow(Fullsize2[:,:,8])
        # plt.colorbar()
        # plt.plot()
        # plt.show()

        Fullsize2 = np.double(morphology.remove_small_objects(Fullsize2, 5))
        # print(Fullsize2)

        # plt.imshow(Fullsize2[:, :, 8])
        # plt.colorbar()
        # plt.plot()
        # plt.show()

        # print(measure.label(Fullsize2).max())

        stack_after = Fullsize2

        y = np.size(Fullsize,0)
        x = np.size(Fullsize,1)
        z = np.size(Fullsize,2)

        stack_after_BW = stack_after.astype(bool)

        stack_after_label, orgnum = measure.label(stack_after, connectivity=1, return_num=True)
        CC = cc3d.connected_components(stack_after_label,connectivity=6)

        # stats1 = measure.regionprops_table(stack_after_label, properties=('label', 'bbox', 'centroid'))
        stats1 = measure.regionprops_table(CC, properties=('label', 'bbox', 'centroid'))


        data = {
            "VoxelList":[[[]]]
        }
        voxels = pd.DataFrame(data)
        # print(f'--------------------------{orgnum}')
        # print(type(voxels.VoxelList[0]))
        for i1 in range(0,512):
            for i2 in range(0,280):
                for i3 in range(0,15):
                    if CC[i1,i2,i3] != 0 :
                        for l in range (1,orgnum+1):
                            if(CC[i1,i2,i3]==l):
                                # print(np.asarray([l, i1, i2, i3]))
                                if(voxels.size<l+1):
                                    voxels.loc[l-1,'VoxelList'] = np.array([[i1,i2,i3]])
                                else:
                                    voxels.loc[l - 1, 'VoxelList'] = np.concatenate(
                                        (np.array(voxels.VoxelList[l - 1]), np.array([[i1, i2, i3]])), axis=0)

        # print(voxels.VoxelList[0])
        print(pd.DataFrame(stats1).size)
        print(voxels.size)

        nib.save(nib.Nifti1Image(np.uint32(stack_after_label),affine=np.eye(4)), addr2 + 'Fullsize_label_' + tt + '.nii')

        niftiwrite(Fullsize2, addr2 + 'Fullsize' + '_' + tt + '.nii')
        print(tt)
        dashline()

        #code to save 3d figure
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.rcParams['figure.dpi'] = 300

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        myCube = np.zeros(shape=(512,280,15))
        for i in range(0,voxels.shape[0]):
            # c1 = [250-i, 100, 100] if i<255 else [110,300-i,110]
            s=str(i+1)
            for j in range(0,np.size(voxels.VoxelList[i],axis=0)):
                myCube[voxels.VoxelList[i][j][0], voxels.VoxelList[i][j][1], voxels.VoxelList[i][j][2]] = i
            ax.text(voxels.VoxelList[i][j][0] + 1, voxels.VoxelList[i][j][1] + 1, voxels.VoxelList[i][j][2] + 1, s,
                    (0, 1, 0), fontsize=5, color = 'red')

        ax.voxels(myCube)



        plt.show()


        fig.savefig( addr2+str(time)+'_3Dconnection2'+'.png')

        niftiwrite(Weights, addr2 + 'Weights_' + tt + '.nii')

    del Fullsize, Fullsize_regression, Fullsize2, Fullsize_input, stats1, \
        orgnum, stack_after, stack_after_BW, stack_after_label, i1, i2, iy, tt