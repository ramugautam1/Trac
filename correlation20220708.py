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
                            # print(np.asarray([l, i1, i2, i3]))
                            if voxels.size < l + 1:
                                voxels.loc[l - 1, 'VoxelList'] = np.array([[i1, i2, i3]])
                            else:
                                voxels.loc[l - 1, 'VoxelList'] = np.concatenate(
                                    (np.array(voxels.VoxelList[l - 1]), np.array([[i1, i2, i3]])), axis=0)

    print(voxels.size)

    for i in range(0, voxels.size):
        dashline()
        print(i)
        if i % 50 == 0:
            print(i/voxels.size)

        if np.size(voxels.VoxelList[i], axis=0) < 30:
            stepsize = 1

        else:
            stepsize = 3

        print(f'stepsize    {stepsize}')

        # for a block of pixels in an object, search for the most correlated nearby block in previous time point
        for n1 in range(0, np.size(voxels.VoxelList[i], axis=0), stepsize):
            if stepsize == 1:
                index = n1
            else:
                # index = min(math.ceil(rd * np.size(voxels.VoxelList[i], axis=0)),np.size(voxels.VoxelList[i])-1)
                index = math.floor(1*rand() * np.size(voxels.VoxelList[i], axis=0))

            # to Feature_map2, copy
            # y component of the pixel - 3 : y component of the pixel + 3,
            # x component of the pixel -3 : x component of the pixel + 3,
            # z component of the pixel -1 : z component of the pixel + 1,
            # all of depth
            # of Fullsize_regression_2_padding

            # print('////////')
            # print(index)
            # print(i)
            # print(np.size(voxels.VoxelList[i],axis=0))

            Feature_map1 = Fullsize_regression_1_padding[
                           voxels.VoxelList[i][index][1]-3:voxels.VoxelList[i][index][1]+3+1,
                           voxels.VoxelList[i][index][0]-3:voxels.VoxelList[i][index][0]+3+1,
                           voxels.VoxelList[i][index][2]-1:voxels.VoxelList[i][index][2]+1+1,
                           :]
            if np.size(Feature_map1,axis=0) < 7 or np.size(Feature_map1,axis=1) < 7 or np.size(Feature_map1, axis=2)<3:
                Feature_map1 = np.pad(Feature_map1, (
                (0, 7 - np.size(Feature_map1, axis=0)), (0, 7 - np.size(Feature_map1, axis=1)),
                (0, 3 - np.size(Feature_map1, axis=2)), (0, 0)))

            # Feature_map1 = Fullsize_regression_1_padding[
            #                voxels.VoxelList[i][index][1] - 3:voxels.VoxelList[i][index][1] + 3,
            #                voxels.VoxelList[i][index][0] - 3:voxels.VoxelList[i][index][0] + 3,
            #                voxels.VoxelList[i][index][2] - 1:voxels.VoxelList[i][index][2] + 1,
            #                :]

            for m1 in range(-1, 2):
                x = 2*m1
                for m2 in range(-1, 2):
                    y = 2*m2
                    for m3 in range(-1, 2):
                        z = m3
                        Feature_map2 = Fullsize_regression_2_padding[
                                       voxels.VoxelList[i][index][1]+x-3:voxels.VoxelList[i][index][1]+x+3+1,
                                       voxels.VoxelList[i][index][0]+y-3:voxels.VoxelList[i][index][0]+y+3+1,
                                       voxels.VoxelList[i][index][2]+z-1:voxels.VoxelList[i][index][2]+z+1+1,
                                       :]
                        # Feature_map2 = Fullsize_regression_2_padding[
                        #                voxels.VoxelList[i][index][1] - 3:voxels.VoxelList[i][index][1] + 3,
                        #                voxels.VoxelList[i][index][0] - 3:voxels.VoxelList[i][index][0] + 3,
                        #                voxels.VoxelList[i][index][2] - 1:voxels.VoxelList[i][index][2] + 1,
                        #                :]

                        if np.size(Feature_map2, axis=0) < 7 or np.size(Feature_map2, axis=1) < 7 or np.size(Feature_map2, axis=2) < 3:
                            Feature_map2 = np.pad(Feature_map2,((0,7-np.size(Feature_map2,axis=0)),(0,7-np.size(Feature_map2,axis=1)),(0,3-np.size(Feature_map2,axis=2)),(0,0)))

                        # dashline()
                        # starline()
                        # print(f'i {i}')
                        # print(f'n1 {n1}')
                        # print(f'index {index}')
                        # print(f'xyz {x} {y} {z}')
                        # print(voxels.VoxelList[i][index][0]+y-3)
                        # print(voxels.VoxelList[i][index][0] + y +1+ 3)
                        # print(np.shape(Feature_map1))
                        # print(np.shape(Feature_map2))
                        # print((Feature_map2.flatten()))
                        # print(np.count_nonzero(np.isnan(Feature_map2)))
                        # dashline()



                        # *****************
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
                        print(f'{i}  {index} {corr}')

                        if corr > 0.2:
                            b = voxels.VoxelList[i]

                            # a = np.zeros(shape=(1,1))
                            a = []
                            for i1 in range(0, np.size(b,axis=0)):
                                # a[i1,0] = Fullsize_1_label[b[i1][1],b[i1][0],b[i1][2]]
                                a.append(Fullsize_1_label[b[i1][1],b[i1][0],b[i1][2]])

                            value = statistics.mode(np.array(a).flatten())

                            u, c = np.unique(np.array(a), return_counts=True)
                            countzero = dict(zip(u, c))[0]



                            if countzero > value:
                                value = 0

                            correlation_map_padding_corr_local = correlation_map_padding_corr[
                                                                 voxels.VoxelList[i][index][1]+x-3:voxels.VoxelList[i][index][1]+x+3+1,
                                                                 voxels.VoxelList[i][index][0]+y-3:voxels.VoxelList[i][index][0]+y+3+1,
                                                                 voxels.VoxelList[i][index][1]+z-1:voxels.VoxelList[i][index][1]+x+1+1]
                            correlation_map_padding_show_local = correlation_map_padding_show[
                                                                 voxels.VoxelList[i][index][1]+x-3:voxels.VoxelList[i][index][1]+x+3+1,
                                                                 voxels.VoxelList[i][index][0]+y-3:voxels.VoxelList[i][index][0]+y+3+1,
                                                                 voxels.VoxelList[i][index][1]+z-1:voxels.VoxelList[i][index][1]+x+1+1]




                            # only select the highest correlation and assign the label
                            correlation_map_padding_show_local[correlation_map_padding_show_local<corr] = value
                            correlation_map_padding_corr_local[correlation_map_padding_corr_local<corr] = corr

                            correlation_map_padding_corr[voxels.VoxelList[i][index][1]+x-3:voxels.VoxelList[i][index][1]+x+3+1,
                                                voxels.VoxelList[i][index][0]+y-3:voxels.VoxelList[i][index][0]+y+3+1,
                                                voxels.VoxelList[i][index][1]+z-1:voxels.VoxelList[i][index][1]+x+1+1] \
                                                = correlation_map_padding_corr_local

                            correlation_map_padding_show[voxels.VoxelList[i][index][1]+x-3:voxels.VoxelList[i][index][1]+x+3+1,
                                                voxels.VoxelList[i][index][0]+y-3:voxels.VoxelList[i][index][0]+y+3+1,
                                                voxels.VoxelList[i][index][1]+z-1:voxels.VoxelList[i][index][1]+x+1+1] \
                                                = correlation_map_padding_show_local




    print(np.amax(correlation_map_padding_show))
    print(addr2 + 'correlation_map_padding_show_traceback' + str(time) + '_' + t2 + '.nii')

    niftiwrite(correlation_map_padding_show,
               addr2+'correlation_map_padding_show_traceback'+str(time)+'_'+t2 +'.nii')

    niftiwrite(correlation_map_padding_corr,
               addr2 + 'correlation_map_padding_hide_traceback' + str(time) + '_' + t2 + '.nii')

    starline()


