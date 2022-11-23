import gc
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
from datetime import datetime
from cycler import cycler

from functions import niftiwrite, dashline, starline, niftiwriteF


def trackStep1(segmentationOutputAddress, trackingOutputAddress, startTime, endTime):

    starline()
    print('                    step 1 begin          ')
    starline()
    tictic = datetime.now()

    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    t1 = startTime
    t2 = endTime
    # t2 = 1
    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [32, 35, I3dw[2]]

    segAddr = segmentationOutputAddress
    trackAddr = trackingOutputAddress
    # for time in range(t1 - 1, t2):

    for time in range(t1, t2 + 1):
        print(f'    time point   {time}')
        tic = datetime.now()
        tt = str(time)

        addr = segAddr + tt + '/'
        addr2 = trackAddr + str(time) + '/'
        print(addr)
        print(addr2)


        # originalImage = np.asarray(nib.load('/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08.nii').dataobj).astype(np.float32).squeeze()

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = sorted(glob.glob(addr + '*.nii'))
        # print(Files1)
        Fullsize = np.zeros(shape=(512, 280, 15))
        Fullsize_input = np.zeros(shape=(512, 280, 15))
        Weights = np.zeros(shape=(512, 280, 15, 64))

        c_file = 0

        for i1 in range(0, I3dw[0], I3d[0]):
            for i2 in range(0, I3dw[1], I3d[1]):

                V_arr = np.asarray(nib.load(Files1[c_file]).dataobj).astype(np.float32).squeeze()
                # print(f'shape {np.shape(V_arr)}')
                # stop
                V_arr = 1 - V_arr
                V2_arr = np.uint8(V_arr * 255)
                # V3_arr = resize(V2_arr, I3d, order=0)
                V3_arr = np.zeros(shape=(32,35,15))
                # print(np.shape(V2_arr[:,:,4]))
                for ix in range(15):
                    V3_arr[:, :, ix] = cv2.resize(V2_arr[:,:,ix], (35, 32), interpolation=cv2.INTER_LINEAR)

                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]
                # print(a, b, c, d)

                # print(f'{a}:{b}, {c}:{d}, : ')

                Fullsize[a:b, c:d, :] = V3_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()

                for iy in range(0, 64):
                    V2_arr = V_arr[:, :, :, iy]
                    V3_arr = resize(V2_arr, I3d, order=0)
                    Weights[a:b, c:d, :, iy] = V2_arr

                V_arr = np.asarray(nib.load(Files1[c_file + 2]).dataobj)#.astype(np.float32).squeeze()
                V3_arr = resize(V_arr, I3d, order=0)
                Fullsize_input[a:b, c:d, :] = V3_arr.squeeze()
                c_file = c_file + 4


        #Remove small itty bitty masks
        Fullsize2 = Fullsize.astype(bool)

        Fullsize2 = np.double(morphology.remove_small_objects(Fullsize2, 60))

        stack_after = Fullsize2

        y = np.size(Fullsize, 0)
        x = np.size(Fullsize, 1)
        z = np.size(Fullsize, 2)

        stack_after_BW = stack_after.astype(bool)

        stack_after_label, orgnum = measure.label(stack_after, connectivity=1, return_num=True)
        CC = cc3d.connected_components(stack_after_label, connectivity=6)

        # stats1 = measure.regionprops_table(stack_after_label, properties=('label', 'bbox', 'centroid'))
        stats1 = pd.DataFrame(measure.regionprops_table(CC, properties=('label', 'bbox', 'centroid', 'coords')))


        # vxllst = stats1.coords
        #
        # Fullsizex_label = np.zeros((512,280,15))
        # for i in range(0,stats1.shape[0]):
        #     for j in range(0,np.size(vxllst[i], axis=0)):
        #         Fullsizex_label[vxllst[i][j][0],vxllst[i][j][1],vxllst[i][j][2]] = stats1.label[i]
        #
        # print(stats1[200:300])
        #
        # niftiwrite(Fullsizex_label,addr2 + 'Fullsizex_label_' + tt + '.nii')


        nib.save(nib.Nifti1Image(np.uint32(stack_after_label), affine=np.eye(4)), addr2 + 'Fullsize_label_' + tt + '.nii')

        niftiwrite(Fullsize2, addr2 + 'Fullsize' + '_' + tt + '.nii')

        # # code to save 3d figure
        # plt.rcParams['figure.figsize'] = (10, 10)
        # plt.rcParams['figure.dpi'] = 500
        # default_cycler = cycler(color=[[1, 0, 0, 0.25], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 1, 1, 0.5]])
        # plt.rc('axes', prop_cycle=default_cycler)
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_box_aspect((350,280,20))


        VoxelList = stats1.coords

        # myCube = np.zeros(shape=(512,280,15))
        # for i in range(0,voxels.shape[0]):
        #     # c1 = [250-i, 100, 100] if i<255 else [110,300-i,110]
        #     s=str(i+1)
        #     for j in range(0,np.size(voxels.VoxelList[i],axis=0)):
        #         myCube[voxels.VoxelList[i][j][0], voxels.VoxelList[i][j][1], voxels.VoxelList[i][j][2]] = i
        #     ax.text(voxels.VoxelList[i][j][0] + 1, voxels.VoxelList[i][j][1] + 1, voxels.VoxelList[i][j][2] + 1, s,
        #             (0, 1, 0), fontsize=5, color = 'red')
        #
        # ax.voxels(myCube)
        Registration = []

        # myCube = np.zeros(shape=(512, 280, 15))


        print('Drawing figure.', end='')
        for i in range(0, VoxelList.shape[0]):
            value = i
            Registration.append([value, stats1['centroid-0'][i], stats1['centroid-1'][i], stats1['centroid-2'][i]])
            if(i%200 == 0):
                print('.', end='')

            # s = str(i+1)
            # for j in range(0, np.size(VoxelList[i], axis=0)):
            #     myCube[VoxelList[i][j][0], VoxelList[i][j][1], VoxelList[i][j][2]] = i
            #
            #
            # # ax.text(VoxelList[i][j][0] + 1, VoxelList[i][j][1] + 1, VoxelList[i][j][2] + 1, s,
            # #    .text(505,280,14, str(VoxelList.shape[0]), (1     (0, 1, 0), fontsize=5, color='red')
            # # ax,1,1), fontsize=10, color='blue')


        #
        # for ww in range(512):
        #     for xx in range(280):
        #         for zz in range(15):
        #             myCube[ww, xx, zz] = originalImage[ww,xx,zz,time,1]

        # ax.voxels(myCube)

        print('\nSaving Files...')
        # plt.show()

        # fig.savefig(addr2 + str(time) + '_3Dconnection2' + '.png')

        niftiwriteF(Weights, addr2 + 'Weights_' + tt + '.nii')

        niftiwriteF(np.array(Registration), addr2 + 'Registration_' + tt + '.nii')

        print('Done.')

        toc = datetime.now()

        print(f'{tt}        completed        time: {toc-tic}')
        dashline()

        # print(gc.get_count())
        # del myCube, Fullsize, Fullsize_regression, Fullsize2, Fullsize_input, Weights, fig, stack_after, stack_after_BW, stack_after_label, tic, toc, Registration, VoxelList, orgnum, CC, addr, addr2
        # gc.collect()
        # print(gc.get_count())



    toctoc = datetime.now()
    print(f'Step 1 completed in {toctoc-tictic}')

    del stats1, orgnum, i1, i2, iy, tt

    gc.collect()