import glob as glob
import cv2
import math as math
import os
import sys
import time
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as scio
from PIL import Image
from skimage.transform import resize
from skimage import measure
from skimage import morphology
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl



def trackStep1():
    # if __name__ == '__main__':
    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    t1 = 1;
    t2 = 41;

    # size of image, size of cuboids
    I3dw = [512, 280, 15]
    I3d = [32, 35, I3dw[2]]
    print(I3d[2])
    # for time in range(t1 - 1, t2):
    for time in range(t1, t2 + 1):
        print(time)
        tt = str(time)

        addr = '/home/nirvan/Desktop/Projects/allFiles/Prediction_Dataset_Ajuba_sqh-cherry_jub-gfp 18A-09/FC-DenseNet/'+str(time) + '/'
        print(addr)
        addr2 = '/home/nirvan/Desktop/Projects/allFiles/' + str(time) + '/'
        print(addr2)

        if not os.path.isdir(addr2):
            os.makedirs(addr2)
        Files1 = sorted(glob.glob(addr + '*.nii'))
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

                print(np.shape(V_arr))

                V_arr = 1 - V_arr
                V2_arr = np.uint8(V_arr * 255)
                V3_arr = resize(V2_arr, I3d, order=1)
                print(np.shape(V3_arr))

                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]

                print(f'a {a}')
                print(f'b {b}')
                print(f'c {c}')
                print(f'd {d}')


                # Fullsize[a:b, c:d, :] = V3
                Fullsize[a:b, c:d, :] = V3_arr

                # V = nib.load(Files1[c_file + 1])
                V_arr = np.asarray(nib.load(Files1[c_file + 1]).dataobj).astype(np.float32).squeeze()

                print(f'V_arr shape {np.shape(V_arr)}')
                for iy in range(0, 64):
                    # print(i2)
                    del V2_arr, V3_arr
                    V2_arr = np.double(V_arr[:, :, :, iy])
                    V3_arr = resize(V2_arr, I3d, order=1)
                    # print(np.shape(V3_arr))
                    Weights[a:b, c:d, :, iy] = V3_arr
                    # print(f'----------------- {iy} ')
                    print(f'{tt}    {i2}   {iy}')

                V_arr = np.asarray(nib.load(Files1[c_file + 2]).dataobj).astype(np.float32).squeeze()
                V3_arr = resize(V_arr, I3d, order=1)
                Fullsize_input[a:b, c:d, :] = V3_arr
                c_file = c_file + 4
                print(np.shape(V3_arr))

        print(sum(Weights[:,:,:,:]))

        #Remove small itty bitty masks
        Fullsize2 = Fullsize.astype(bool)

        # for it in range(0, np.size(Fullsize,2)):
        #     img = Fullsize2[:,:,it]
        #     f = measure.label(img)
        #     orgnum = f.max()
        #     print(f'{f} {orgnum}')
        #     g = measure.regionprops(f, 'area')
        #     print('--0--00-0-0-0-0-0-0-0-0-0-0---0-0-0-0-0-0')
        #     print(g)
        #     print('--0--00-0-0-0-0-0-0-0-0-0-0---0-0-0-0-0-0')
        #
        # #
        plt.imshow(Fullsize2[:,:,8])
        plt.colorbar()
        plt.plot()
        plt.show()

        Fullsize2 = np.double(morphology.remove_small_objects(Fullsize2, 5))
        print(Fullsize2)

        plt.imshow(Fullsize2[:, :, 8])
        plt.colorbar()
        plt.plot()
        plt.show()

        print(Fullsize2.max())
        print(measure.label(Fullsize2).max())
        print("Shape")
        print(np.shape(Fullsize2))

        stack_after = Fullsize2

        y = np.size(Fullsize,0)
        x = np.size(Fullsize,1)
        z = np.size(Fullsize,2)

        stack_after_BW = stack_after.astype(bool)
        stack_after_label, orgnum = measure.label(stack_after_BW, connectivity=1, return_num=True)
        # orgnum = stack_after_label.max()
        print(f'--------xxxx--------xxxx--------{orgnum}')
        stats = measure.regionprops_table(stack_after_label, properties=('label', 'bbox', 'centroid'))
        print(pd.DataFrame(stats))

        stack_after_label, orgnum = measure.label(stack_after, connectivity=1, return_num=True)
        print(f'------{orgnum}')
        stats1 = measure.regionprops_table(stack_after_label, properties=('label', 'bbox', 'centroid'))
        print(pd.DataFrame(stats1))
        print(orgnum)
        nib.save(nib.Nifti1Image(np.uint32(stack_after_label),affine=np.eye(4)), addr2 + 'Fullsize_label_' + tt + '.nii')
        # new_image = nib.Nifti1Image(data, affine=np.eye(4))
        stop







