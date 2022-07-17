import math

import numpy as np
import pandas as pd
from skimage import measure
import statistics

from functions import rand, size3, niftiwrite, niftiread, dashline, starline


def correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2,
                t2, time, spatial_extend_matrix, addr2, padding):

    depth = np.size(Fullsize_regression_1, axis=3)

    # get the size of sample
    [x, y, z] = size3(Fullsize_1)
    [x_reserve, y_reserve, z_reserve] = size3(Fullsize_1)
    print([x, y, z])

    #padding the sample for 'extended search' (Fullsize: object label map, Fullsize_regression: object deep feature map)
    Fullsize_1_padding = np.pad(Fullsize_1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
    Fullsize_2_padding = np.pad(Fullsize_2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2])), 'constant')
    Fullsize_regression_1_padding = np.pad(Fullsize_regression_1, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]), (0, 0)), 'constant')
    Fullsize_regression_2_padding = np.pad(Fullsize_regression_2, ((padding[0], padding[0]), (padding[1], padding[1]), (padding[2], padding[2]), (0, 0)), 'constant')


    # correlation_map_padding=zeros(x+padding*2, y+padding*2, z+4,max(max(max(Fullsize_1))));
    correlation_map_padding_corr = np.zeros((x+padding[0]*2, y+padding[1]*2, z+padding[2]*2))
    correlation_map_padding_show = np.zeros((x+padding[0]*2, y+padding[1]*2, z+padding[2]*2))

    del Fullsize_regression_1, Fullsize_regression_2, Fullsize_1, Fullsize_2

    Fullsize_1_label = Fullsize_1_padding

    print(np.amax(Fullsize_1_padding))


    labelss, orgnum = measure.label(Fullsize_1_padding, connectivity=1, return_num=True)
    print(orgnum)


    [fx, fy, fz] = size3(Fullsize_1_padding)

    data = {
        "VoxelList": [[[]]]
    }
    voxels = pd.DataFrame(data)
    # print(f'--------------------------{orgnum}')
    # print(type(voxels.VoxelList[0]))
    for i1 in range(0, fx):
        for i2 in range(0, fy):
            for i3 in range(0, fz):
                if Fullsize_1_padding[i1, i2, i3] != 0:
                    for l in range(1, orgnum+1):
                        if Fullsize_1_padding[i1, i2, i3] == l:
                            print(l)
                            if voxels.size < l + 1:
                                voxels.loc[l - 1, 'VoxelList'] = np.array([[i1, i2, i3]])
                            else:
                                voxels.loc[l - 1, 'VoxelList'] = np.concatenate(
                                    (np.array(voxels.VoxelList[l - 1]), np.array([[i1, i2, i3]])), axis=0)

    print(voxels.size)

    for i in range(0, voxels.size):
        dashline()

        if np.size(voxels.VoxelList[i], axis=0) < 30:
            stepsize = 1

        else:
            stepsize = 3

        print(f'i {i}    stepsize    {stepsize}')

        # for a block of pixels in an object, search for the most correlated nearby block in previous time point
        for n1 in range(0, np.size(voxels.VoxelList[i], axis=0), stepsize):
            if stepsize == 1:
                index = n1
            else:
                index = math.ceil(rand() * np.size(voxels.VoxelList[i], axis=0))
            if index >= np.size(voxels.VoxelList[i], axis=0):
                index = np.size(voxels.VoxelList[i], axis=0) - 1

            # to Feature_map2, copy ==> y component of the pixel - 3 : y component of the pixel + 3, x component of the pixel -3 : x component of the pixel + 3,
                        #                z component of the pixel -1 : z component of the pixel + 1, all of depth <==of Fullsize_regression_2_padding

            print(f'i {i} index {index} vox size {np.size(voxels.VoxelList[i],axis=0)}')

            Feature_map1 = np.copy(Fullsize_regression_1_padding[
                           voxels.VoxelList[i][index][0]-3:voxels.VoxelList[i][index][0]+3+1,
                           voxels.VoxelList[i][index][1]-3:voxels.VoxelList[i][index][1]+3+1,
                           voxels.VoxelList[i][index][2]-1:voxels.VoxelList[i][index][2]+1+1,
                           :])
    #=================================================
            # if np.size(Feature_map1,axis=0) < 7 or np.size(Feature_map1,axis=1) < 7 or np.size(Feature_map1, axis=2)<3:
                # print(np.shape(Feature_map1))
                # break
                # Feature_map1 = np.pad(Feature_map1, (
                # (0, 7 - np.size(Feature_map1, axis=0)), (0, 7 - np.size(Feature_map1, axis=1)),
                # (0, 3 - np.size(Feature_map1, axis=2)), (0, 0)))


            for m1 in range(-1, 2):
                x = 2*m1
                for m2 in range(-1, 2):
                    y = 2*m2
                    for m3 in range(-1, 2):
                        z = m3
                        Feature_map2 = np.copy(Fullsize_regression_2_padding[
                                       voxels.VoxelList[i][index][0]+x-3:voxels.VoxelList[i][index][0]+x+3+1,
                                       voxels.VoxelList[i][index][1]+y-3:voxels.VoxelList[i][index][1]+y+3+1,
                                       voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1,
                                       :])
                        # Feature_map2 = Fullsize_regression_2_padding[
                        #                voxels.VoxelList[i][index][1] - 3:voxels.VoxelList[i][index][1] + 3,
                        #                voxels.VoxelList[i][index][0] - 3:voxels.VoxelList[i][index][0] + 3,
                        #                voxels.VoxelList[i][index][2] - 1:voxels.VoxelList[i][index][2] + 1,
                        #                :]
#==============================================
                        # if np.size(Feature_map2, axis=0) < 7 or np.size(Feature_map2, axis=1) < 7 or np.size(Feature_map2, axis=2) < 3:
                            # print(np.shape(Feature_map2))
                            # break
                            # Feature_map2 = np.pad(Feature_map2,((0,7-np.size(Feature_map2,axis=0)),(0,7-np.size(Feature_map2,axis=1)),(0,3-np.size(Feature_map2,axis=2)),(0,0)))


                        # *****************left to do
                        # ---uncomment if the extended search decay is wanted
                        # Feature_map2=Feature_map2.*spatial_extend_matrix;
                        # Feature_map1=Feature_map1/mean2(Feature_map1);
                        # Feature_map2=Feature_map2/mean2(Feature_map2);
                        # corr = convn(Feature_map1,Feature_map2(end:-1:1,end:-1:1,end:-1:1));

                        # # Flattening the feature map
                        Feature_map1_flatten = Feature_map1.flatten(order='F')
                        Feature_map2_flatten = Feature_map2.flatten(order='F')

                        #calculate correlation
                        corr = np.corrcoef(Feature_map1_flatten, Feature_map2_flatten)[0,1]
                        # print(f'{i}  {index} {corr}')

                        if corr > 0.2:
                            b = voxels.VoxelList[i]
                            print(corr)
                            # print(b)

                            # a = np.zeros(shape=(1,1))
                            a = []
                            for i1 in range(0, np.size(b,axis=0)):
                                print(Fullsize_1_label[b[i1][0], b[i1][1], b[i1][2]])
                                a.append(Fullsize_1_label[b[i1][0], b[i1][1], b[i1][2]])
                                # print(Fullsize_1_label[b[i1][1],b[i1][0],b[i1][2]])
                                # a.append(Fullsize_1_label[b[i1][1],b[i1][0],b[i1][2]])

                            value = statistics.mode(np.array(a).flatten())

                            u, c = np.unique(np.array(a), return_counts=True)
                            try:
                                countzero = dict(zip(u, c))[0]
                            except:
                                countzero = 0

                            print(f'a-shape: {np.shape(a)} countzero {countzero} value {value}')
                            print(a)



                            if countzero > value:
                                value = 0

                            correlation_map_padding_corr_local = correlation_map_padding_corr[
                                                                 voxels.VoxelList[i][index][0]+x-3:voxels.VoxelList[i][index][0]+x+3+1,
                                                                 voxels.VoxelList[i][index][1]+y-3:voxels.VoxelList[i][index][1]+y+3+1,
                                                                 voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1]

                            correlation_map_padding_show_local = correlation_map_padding_show[
                                                                 voxels.VoxelList[i][index][0]+x-3:voxels.VoxelList[i][index][0]+x+3+1,
                                                                 voxels.VoxelList[i][index][1]+y-3:voxels.VoxelList[i][index][1]+y+3+1,
                                                                 voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1]




                            # only select the highest correlation and assign the label

                            correlation_map_padding_show_local[correlation_map_padding_show_local<corr] = value
                            correlation_map_padding_corr_local[correlation_map_padding_corr_local<corr] = corr

                            correlation_map_padding_corr[voxels.VoxelList[i][index-1][0]+x-3:voxels.VoxelList[i][index-1][0]+x+3+1,
                                                voxels.VoxelList[i][index][1]+y-3:voxels.VoxelList[i][index][1]+y+3+1,
                                                voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1] \
                                                = correlation_map_padding_corr_local

                            correlation_map_padding_show[voxels.VoxelList[i][index][0]+x-3:voxels.VoxelList[i][index][0]+x+3+1,
                                                voxels.VoxelList[i][index][1]+y-3:voxels.VoxelList[i][index][1]+y+3+1,
                                                voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1] \
                                                = correlation_map_padding_show_local
                            print('cor map pad show local shape')
                            print(np.shape(correlation_map_padding_show_local))





    print(np.amax(correlation_map_padding_show))
    print(addr2 + 'correlation_map_padding_show_traceback' + str(time) + '_' + t2 + '.nii')

    niftiwrite(correlation_map_padding_show,
               addr2+'correlation_map_padding_show_traceback'+str(time)+'_'+t2 +'.nii')

    niftiwrite(correlation_map_padding_corr,
               addr2 + 'correlation_map_padding_hide_traceback' + str(time) + '_' + t2 + '.nii')

    starline()


