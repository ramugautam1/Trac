import statistics

import scipy.io as scio
import numpy as np
import pandas as pd
from functions import line
from datetime import datetime
import xlsxwriter
import os
import math
import glob as glob
import nibabel as nib
from skimage import measure

from correlation20220708 import correlation
# from testCorr import correlation
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand, nan_2d


def trackStep2():
    starline()  # print **************************************
    print('step 2 start')
    starline()
    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    I3dw = [512, 280, 15]
    padding = [20, 20, 2]
    timm = datetime.now()

    folder = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08_Tracking_Result/'
    trackbackT = 2

    if not os.path.isdir(folder):
        print(os.makedirs(folder))

    filename = folder + 'TrackingID' + str(timm) + '.xlsx'  # the excel file name to write the tracking result

    workbook = xlsxwriter.Workbook(filename)

    worksheet1 = workbook.add_worksheet()
    worksheet2 = workbook.add_worksheet()
    worksheet3 = workbook.add_worksheet()
    worksheet4 = workbook.add_worksheet()
    worksheet5 = workbook.add_worksheet()
    worksheet6 = workbook.add_worksheet()
    worksheet7 = workbook.add_worksheet()
    worksheet8 = workbook.add_worksheet()
    worksheet9 = workbook.add_worksheet()
    worksheet10 = workbook.add_worksheet()
    worksheet11 = workbook.add_worksheet()
    worksheet12 = workbook.add_worksheet()

    worksheet2.write('A1', 'time')  # write titles to excel
    worksheet2.write('B1', 'old')
    worksheet2.write('C1', 'new')
    worksheet2.write('D1', 'split')
    worksheet2.write('E1', 'fusion')

    print('Excel file created.')

    depth = 64  # the deep features to take in correlation calculation
    initialpoint = 1  # the very first time point of all samples
    startpoint = 1  # the time point to start tracking
    endpoint = 41  # the time point to stop tracking

    # xlswriter1 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter2 = pd.DataFrame(np.zeros((endpoint, 5)))
    # xlswriter3 = nan_2d(20000, endpoint * 2)
    # xlswriter4 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter5 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter6 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter7 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter8 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter9 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter10 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter11 = pd.DataFrame(nan_2d(20000, endpoint * 2))
    # xlswriter12 = pd.DataFrame(nan_2d(20000, endpoint * 2))

    print(f'depth = {depth}, startpoint = {startpoint}, endpoint = {endpoint}')

    spatial_extend_matrix = np.full((10, 10, 3, depth),
                                    0)  # the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1 + 1 - 5) + (i2 + 1 - 5) + (i3 + 1 - 2)) / 20)

    for time in range(startpoint, endpoint + 1):
        dashline()
        tic = datetime.now()
        print('time point: ' + str(time))
        t1 = str(time)
        t2 = str(time + 1)
        worksheet1.write(0, time * 2 - 2, str(t1))
        worksheet1.write(0, time * 2 - 1, str(t2))
        worksheet3.write(0, time * 2 - 1, str(t2))
        worksheet4.write(0, time * 2 - 1, str(t2))
        worksheet5.write(0, time * 2 - 1, str(t2))
        worksheet6.write(0, time * 2 - 1, str(t2))
        worksheet7.write(0, time * 2 - 1, str(t2))
        worksheet8.write(0, time * 2 - 1, str(t2))
        worksheet9.write(0, time * 2 - 1, str(t2))
        worksheet10.write(0, time * 2 - 1, str(t2))
        worksheet11.write(0, time * 2 - 1, str(t2))
        worksheet12.write(0, time * 2 - 1, str(t2))

        addr1 = folder + t1 + '/'
        addr2 = folder + t2 + '/'
        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)
        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)
        Files1 = sorted(glob.glob(addr1 + '*.nii'))
        Files2 = sorted(glob.glob(addr2 + '.nii'))

        # calculate correlation between this and next time point, using (labeled images and weights from step 1)

        if time - initialpoint < trackbackT:  # calculating correlation for start time points (e.g. time=2)
            for i1 in range(1, time - initialpoint + 1 + 1):
                print(f'time  point: {time}')
                print(addr2)
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                if i1 == time - initialpoint + 1:
                    print(addr1)
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_label_' + t1 + '.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
                else:
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')

                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                            spatial_extend_matrix, addr2, padding)
                dashline()

        else:
            for i1 in range(1, trackbackT + 1):
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_' + t1 + '.nii')
                Fullsize_regression_1 = niftiread(addr1 + 'Weights_' + t1 + '.nii')
                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1, Fullsize_regression_2, t2, i1,
                            spatial_extend_matrix, addr2, padding)
                dashline()

        # del Fullsize_1, Fullsize_regression_1,Fullsize_2, Fullsize_regression_2#, Fullsize_1_padding, Fullsize_2_padding, \
        # Fullsize_regression_1_padding, Fullsize_regression_2_padding, Fullsize_1_label, Fullsize_2_label

        # plot tracking
        t1 = str(time)
        t2 = str(time + 1)

        # read the correlation calculation results
        correlation_map_padding_show1 = niftiread(
            folder + t2 + '/' + 'correlation_map_padding_show_traceback1_' + t2 + '.nii')
        correlation_map_padding_hide1 = niftiread(
            folder + t2 + '/' + 'correlation_map_padding_hide_traceback1_' + t2 + '.nii')

        # Reading centroids

        if time - initialpoint < trackbackT and time > initialpoint:
            for i1 in range(1, time - initialpoint + 1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1) + '_' + t2 + '.nii')

                for i2 in range(0, I3dw[0] + padding[0] * 2):  # 0 because python
                    for i3 in range(0, I3dw[1] + padding[1] * 2):
                        for i4 in range(0, I3dw[2] + padding[2] * 2):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[
                                i2, i3, i4] and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2, i3, i4]
        elif time - initialpoint >= trackbackT and time > initialpoint:
            for i1 in range(2, trackbackT + 1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(
                    folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1) + '_' + t2 + '.nii')
                for i2 in range(0, I3dw[0] + padding[0] * 2):  # 0 because python
                    for i3 in range(0, I3dw[1] + padding[1] * 2):
                        for i4 in range(0, I3dw[2] + padding[2] * 2):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[
                                i2, i3, i4] and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2, i3, i4]
        else:
            Registration1 = niftiread(folder + t1 + '/' + 'Registration_' + t1 + '.nii')
            # print('here')
        # -----------------------------------------------------------Good Until Here---------------------------------------------------------------------------------------------
        # Read segmentation
        Fullsize_2 = niftiread(folder + t2 + '/Fullsize_label_' + t2 + '.nii').astype(int)
        Fullsize_2_2 = np.zeros(shape=(np.shape(Fullsize_2)))

        # crop the expanded sample to its original size
        correlation_map_padding_show2 = correlation_map_padding_show1[20:-1 * padding[0], 20:-1 * padding[1],
                                        2:-1 * padding[2]]
        Fullsize_2_mark = correlation_map_padding_show2

        if time > initialpoint:
            correlation_map_padding_show2_2 = correlation_map_padding_show1_2[20:-1 * padding[0], 20:-1 * padding[1],
                                              2:-1 * padding[2]]
            Fullsize_1 = correlation_map_padding_show2_2
            Fullsize_1[Fullsize_1 == 0] = np.nan

            # if not initial time point, read the fusion data of last time point (saved in the same folder as this time point, t1)

            # detector_fusion_old=load(strcat('D:\NEW\',folder,'\',t1,'\fusion_tracking_',t1,'.mat'),'detector3_fusion');
            # for i1=2:2:size(detector_fusion_old.detector3_fusion,1)
            #     detector_fusion_old.detector3_fusion(i1,:)=0;
            # end

            # ------------------------------------------------------------------

            detector_fusion_old = scio.loadmat(folder + t1 + '/' + 'fusion_tracking_' + t1 + '.mat', 'detector3_fusion')
            for i1 in range(1, np.size(detector_fusion_old.detector3_fusion, axis=0) + 1, 2):
                detector_fusion_old.detector3_fusion[i1, :] = 0

            Fullsize_2_mark[Fullsize_2 == 0] = 0
            # ------------------------------------------------------------------

        # del correlation_map_padding_show1, correlation_map_padding_show1_2, correlation_map_padding_hide1, correlation_map_padding_hide1_2

        # Get the object characteristics
        Fullsize_2_mark_BW = Fullsize_2_mark
        Fullsize_2_mark_BW[Fullsize_2_mark_BW > 0] = 1
        Fullsize_2_mark_BW = Fullsize_2_mark_BW.astype(bool)
        Fullsize_2_mark_label, orgnum = measure.label(Fullsize_2_mark, connectivity=1, return_num=True)

        # stats1 = regionprops3(Fullsize_2,'BoundingBox','VoxelList','ConvexHull','Centroid');
        stats1 = pd.DataFrame(
            measure.regionprops_table(Fullsize_2.astype(int), properties=('label', 'bbox', 'coords', 'centroid')))
        VoxelList = stats1.coords

        #  sort the objects in descending order of size

        count = np.zeros(stats1.shape[0])
        for i in range(stats1.shape[0]):
            count[i] = np.size(stats1.coords[i], axis=0)
        stats1['Count'] = count.astype(int)

        stats2 = stats1.sort_values(by='Count', axis=0, ascending=False, ignore_index=False)
        print(f'object count {np.amax(stats2.label)}')

        # -----
        detector_fusion = []
        detector_split = {}
        detector2_fusion = []
        detector3_fusion = []

        # stack_after_label[Fullsize_2_mark > 0] = 0  Not used, not initialized

        Fullsize_2_mark[Fullsize_2_mark == 0] = np.nan

        # Initialize new Registration variables
        newc = -1
        l = np.size(Registration1, axis=0)
        Registration2 = {}
        detector_old = {}
        detector_new = {}
        detector_numbering = {}
        c1 = 0
        c2 = 0
        c3 = 0
        c_numbering = -1
        cc = {}

        # Tracking each object
        xlswriter1 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter2 = pd.DataFrame(np.zeros((endpoint, 5)))
        xlswriter3 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter4 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter5 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter6 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter7 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter8 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter9 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter10 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter11 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        xlswriter12 = pd.DataFrame(nan_2d(20000, endpoint * 2))
        print(stats2.shape[0])
        for i in range(stats2.shape[0]):
            print(f'Obj No. {i+1} (i)')
            max_object_intensity1 = 0
            max_object_intensity2 = 0
            b = stats2.coords[i]
            if time + 1 < 10:
                ttag = '00'
            elif time + 1 < 100:
                ttag = '0'
            else:
                ttag = ''

            threeDimg1 = niftiread(
                '/home/nirvan/Desktop/Projects/EcadMyo_08_all/3DImage/' + 'EcadMyo_08/' + 'Ecad/' + 'threeDimg_' +
                ttag + str(time + 1) + '.nii')
            threeDimg2 = niftiread(
                '/home/nirvan/Desktop/Projects/EcadMyo_08_all/3DImage/' + 'EcadMyo_08/' + 'Myo/' + 'threeDimg_' +
                ttag + str(time + 1) + '.nii')

            threeDimgPixelList1 = {}
            threeDimgPixelList2 = {}

            for i1 in range(np.size(b, axis=0)):
                threeDimgPixelList1[i1] = threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]]
                threeDimgPixelList2[i1] = threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]]

                if threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]] > max_object_intensity1:
                    max_object_intensity1 = threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]]

                if threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]] > max_object_intensity1:
                    max_object_intensity2 = threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]]

            threeDimgPixelList1 = sorted(threeDimgPixelList1, reverse=True)
            threeDimgPixelList2 = sorted(threeDimgPixelList2, reverse=True)

            # Average the pixels to get average object intensity
            average_object_intensity1 = sum(threeDimgPixelList1) / np.size(b, axis=0)
            average_object_intensity2 = sum(threeDimgPixelList2) / np.size(b, axis=0)

            a = {}
            a_t_1 = {}
            # k = boundary(b)                                                                         ### RRR
            print(f'       a      {a}')

            for i1 in range(np.size(b, axis=0)):
                # print(b[i1, 0], b[i1, 1], b[i1, 2])
                # print(Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]])
                a[i1] = Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]]

            datta = np.array(list(a.values()))

            value = statistics.mode(np.array(datta).flatten())

            if (np.isnan(value)):
                print('-------nan--------')

            # u, cx = np.unique(np.array(datta), return_counts=True)
            #
            # Value_f = dict(zip(u, cx))[value]
            #
            # countnan = a.count(np.nan)
            #
            # if countnan > Value_f:
            #     value = np.nan

            if time > startpoint:
                for i1 in range(np.size(b, axis=0)):
                    a_t_1[i1] = Fullsize_1[b[i1, 0], b[i1, 1], b[i1, 2]]
                # ----
                datta = np.array(list(a.values_t_1))
                if np.count_nonzero(np.isnan(datta)) == len(a_t_1):
                    Value_f_t_1 = np.nan
                else:
                    value_t_1 = statistics.mode(datta[~np.isnan(datta)].flatten())
                    u, cx = np.unique(np.array(a_t_1), return_counts=True)
                    Value_f_t_1 = dict(zip(u, cx))[value_t_1]
                # ----
                print(f'Value_f_t_1 {Value_f_t_1}')
                # Check whether the object has already merged in the last time point
                if not np.isnan(value_t_1) and isempty(intersect(value_t_1, np.array(Registration1.keys()))) \
                        and not isempty(intersect(value_t_1, detector_fusion_old.detector3_fusion)):  # merge happened in last time point
                    detector_numbering[value] = value_t_1
                    value = value_t_1
                    c_numbering = c_numbering + 1
                    print(value)

            print(f'Registration2 {Registration2} \nkeys:')
            print(np.array(list(Registration2.keys())))
            # Check whether the object has already been tracked in the current time point
            if not isempty(intersect(value, np.array(list(Registration2.values())))):
                # print(Registration2)
                print(f'{value}------in-----{a}')
                value2 = setdiff(np.array(list(a.values())), np.array(list(Registration2.values())))
                print(f'value2 {value2}')
                print(not isempty(value2))
                print(np.size(value2))

                if not isempty(value2) and np.size(value2) > 0 and not isempty(intersect(value2, Registration1[:, 0])):
                    value = value2[0, math.ceil(rand() * np.size(value2, axis=1))]

            # If the representatiove of an object is NaN, it means that it is a new object, Assign new ID to it
            if not np.isnan(value):
                value = int(value)
            if np.isnan(value):
                color = [0, 0, 0]
                newc += 1
                Registration2[l+newc] = [l+newc,stats2['centroid-0'][i], stats1['centroid-1'][i], stats1['centroid-2'][i]]
                value = l + newc
                # Reassign new labels to the sample
                for i1 in range(np.size(b, axis=0)):
                    Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]] = value
                    Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = value

                txt = 'NEW ' + str(value)
                detector_new[c1] = (value)
                c1 += 1


                # Document the object characteristics
                xlswriter1.iloc[newc, time * 2 - 2] = 'new'
                xlswriter1.iloc[newc, time * 2 - 1] = str(newc)

                xlswriter3.iloc[newc, time * 2 - 1] = str(max_object_intensity1)
                xlswriter4.iloc[newc, time * 2 - 1] = str(average_object_intensity1)
                xlswriter5.iloc[newc, time * 2 - 1] = str(np.size(b, axis=0))

                xlswriter6.iloc[newc, time * 2 - 1] = str(stats2['centroid-0'][i])
                xlswriter7.iloc[newc, time * 2 - 1] = str(stats2['centroid-1'][i])
                xlswriter8.iloc[newc, time * 2 - 1] = str(stats2['centroid-2'][i])

                xlswriter11.iloc[newc, time * 2 - 1] = str(max_object_intensity2)
                xlswriter12.iloc[newc, time * 2 - 1] = str(average_object_intensity2)

                # draw_text(value) = text(b(end, 1), b(end, 2), b(end, 3), txt, 'Rotation', +15)              # RRR

            # if the representative is not a NaN, it means we find a tracking
            elif not np.isnan(value) and value > 0:
                if isempty(intersect(value, np.array(list(Registration2.keys())))):
                    # color = map(value,1:3)                                                                  # RRR

                    # Registration2[value][0] = value
                    Registration2[value] = [l+newc, stats2['centroid-0'][i], stats2['centroid-1'][i],
                                                 stats2['centroid-2'][i]]

                    txt = 'OLD ' + str(value)

                    # Reassign new labels to the sample
                    for i1 in range(np.size(b, axis=0)):
                        Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = value

                    detector_old[c2] = value


                    print(f'value {value} time {time}')
                    c2 += 1
                    tx = time*2-2
                    xlswriter1.iloc[value, tx] = str(value)
                    xlswriter1.iloc[value, time * 2 - 1] = str(value)
                    xlswriter3.iloc[newc, time * 2 - 1] = str(max_object_intensity1)

                    xlswriter4.iloc[newc, time * 2 - 1] = str(average_object_intensity1)
                    xlswriter5.iloc[newc, time * 2 - 1] = str(np.size(b, axis=0))

                    xlswriter6.iloc[newc, time * 2 - 1] = str(stats2['centroid-0'][i])
                    xlswriter7.iloc[newc, time * 2 - 1] = str(stats2['centroid-1'][i])
                    xlswriter8.iloc[newc, time * 2 - 1] = str(stats2['centroid-2'][i])

                    xlswriter11.iloc[newc, time * 2 - 1] = str(max_object_intensity2)
                    xlswriter12.iloc[newc, time * 2 - 1] = str(average_object_intensity2)

                    draw_forsure = 0  #### RRR

                #
                # Draw code is here. I'm confused
                #

                # If the representative is not NaN but is zero, it indicates a split
                else:
                    # color = map(value,1:3)
                    newc += 1
                    Registration2[l+newc]= [value, stats2['centroid-0'][i], stats2['centroid-1'][i],
                                                stats2['centroid-2'][i]]
                    detector_split[value]=l+newc

                    xlswriter10.iloc[value, time * 2 - 1] = str(value)
                    xlswriter10.iloc[newc, time * 2 - 1] = str(value)
                    # Reassign new labels to the sample
                    for i1 in range(np.size(b, axis=1)):
                        Fullsize_2_2[b[i1, 0], b[i1, 1], b[i1, 2]] = newc
                        Fullsize_2_mark[b[i1, 0], b[i1, 1], b[i1, 2]] = value

                    for ix in range((time - 1) * 2):
                        xlswriter1.iloc[newc, ix] = xlswriter1.iloc[value, ix]
                    var = (time - 1) * 2 - 1
                    xlswriter1.iloc[newc, var] = str(value)
                    xlswriter1.iloc[newc, var] = str(newc)
                    xlswriter3.iloc[newc, var] = str(max_object_intensity1)
                    xlswriter4.iloc[newc, var] = str(average_object_intensity1)
                    xlswriter5.iloc[newc, var] = str(np.size(b, axis=0))
                    xlswriter6.iloc[newc, var] = str(stats2['centroid-0'][i])
                    xlswriter7.iloc[newc, var] = str(stats2['centroid-1'][i])
                    xlswriter8.iloc[newc, var] = str(stats2['centroid-2'][i])
                    xlswriter11.iloc[newc, var] = str(max_object_intensity2)
                    xlswriter12.iloc[newc, var] = str(average_object_intensity2)

                    for i2 in range(time * 2):
                        if xlswriter1.iloc[value, i2] != np.nan and str(xlswriter1.iloc[value, i2] != "new"):
                            value = str(xlswriter1.iloc[value, i2])
                            break

                    value = newc
        print(xlswriter1)
        print(xlswriter2)
        # colormap(map)             RRR
        print(Registration2)

        # Write the tracking result
        niftiwriteF(np.array(list(Registration2.values())), addr2 + 'Registration2_tracking_' + t2 + '.nii')
        niftiwriteF(Fullsize_2_2, addr2 + 'Fullsize_2_aftertracking_' + t2 + '.nii')
    #
    #     # Tracking old and split object is almost done, now time for fusion detection and alarms
    #
    #     c = 0
    #
    #     for i1 in range(stats2.shape[0]):  # for i1=1:size(stats2.VoxelList,1)
    #         b = stats2.coords[i1]
    #
    #         UNIQUEcount = []
    #
    #         for i2 in range(np.size(b, axis=0)):
    #             UNIQUEcount.append(Fullsize_2_mark[b[i2, 0], b[i2, 1], b[i2, 2]])
    #             if np.isnan(np.array(UNIQUEcount)[i2]):
    #                 UNIQUEcount[i2, 0] = 0
    #
    #         uniq, cnts = np.unique(np.array(UNIQUEcount), return_counts=True)
    #         value_counts = np.row_stack((uniq, cnts))
    #
    #         if np.size(value_counts, axis=0) > 1:  # if length(C) > 1
    #             for j in range(2):
    #                 detector_fusion.append(value_counts[j])
    #             # c += 2
    #
    #     for i1 in range(0, np.size(detector_fusion, axis=0), 2):  # fusion 0 filter
    #         print(np.shape(detector_fusion))
    #
    #         if np.array(detector_fusion)[i1, 0] == 0:
    #             for iy in range(i1, i1 + 2):
    #                 detector_fusion[iy][0:-1] = detector_fusion[iy][1:]
    #
    #     detector2_fusion = detector_fusion
    #
    #     for i1 in range(0, np.size(detector2_fusion, axis=0), 2):
    #         for i2 in range(0, np.size(detector2_fusion, axis=1)):
    #             if intersect(detector2_fusion[i1, i2], Registration2[:, 1]):
    #                 for ix in range(i1, i1 + 1 + 1):
    #                     detector2_fusion[ix, i2] = 0
    #         for i2 in range(0, np.size(detector2_fusion, axis=1), 1):
    #             if detector2_fusion[i1 + 1, i2] < 5:
    #                 for ix in range(i1, i1 + 1 + 1):
    #                     detector2_fusion[ix, i2] = 0
    #
    #     c = 0
    #     detector3_fusion = []
    #     for i1 in range(0, np.size(detector2_fusion, axis=0), 2):
    #         if np.count_nonzero(detector2_fusion[i1, :]) != 0:
    #             for cx in range(2):
    #                 detector3_fusion.append(detector_fusion[i1 + cx, :])
    #                 c += 2  # not used here
    #
    #     # Replace the labels and colors inthe figure based on object identified
    #     # Figure code missing
    #
    #     for i2 in range(0, np.size(detector3_fusion, axis=0), 2):
    #         detector3_fusion_exist = 0;
    #         for i1 in range(np.size(detector3_fusion, axis=1)):
    #             if detector3_fusion[i2, i1] > 0:
    #                 None
    #
    # # npz, pickle !!!!!!!!!!!!!!!
    #
    # #
    #
    # writer = pd.ExcelWriter('filename', engine='xlsxwriter')
    #
    # xlswriter1.to_excel(writer, sheet_name='Sheet1', startrow=0)
    # xlswriter2.to_excel(writer, sheet_name='Sheet2', startrow=1)
    # xlswriter3.to_excel(writer, sheet_name='Sheet3', startrow=0)
    # xlswriter4.to_excel(writer, sheet_name='Sheet4', startrow=0)
    # xlswriter5.to_excel(writer, sheet_name='Sheet5', startrow=0)
    # xlswriter6.to_excel(writer, sheet_name='Sheet6', startrow=0)
    # xlswriter7.to_excel(writer, sheet_name='Sheet7', startrow=0)
    # xlswriter8.to_excel(writer, sheet_name='Sheet8', startrow=0)
    # xlswriter9.to_excel(writer, sheet_name='Sheet9', startrow=0)
    # xlswriter10.to_excel(writer, sheet_name='Sheet10', startrow=0)
    # xlswriter11.to_excel(writer, sheet_name='Sheet11', startrow=0)
    # xlswriter12.to_excel(writer, sheet_name='Sheet12', startrow=0)

    # workbook.close()

# USE DICTIONARY!!!!!!!!!!!!!!!!!!!!!
