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
from functions import dashline, starline, niftiread, niftiwrite, niftiwriteF, intersect, setdiff, isempty, rand


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

    spatial_extend_matrix = np.full((10, 10, 3, depth), 0)  # the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1+1-5)+(i2+1-5)+(i3+1-2))/20)

    for time in range(startpoint, endpoint+1):
        dashline()
        tic = datetime.now()
        print('time point: ' + str(time))
        t1 = str(time)
        t2 = str(time+1)
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
        Files2 = sorted(glob.glob(addr2+'.nii'))

        if time-initialpoint < trackbackT:  # calculating correlation for start time points (e.g. time=2)
            for i1 in range(1, time-initialpoint+1+1):
                print(f'time  point: {time}')
                print(addr2)
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                if i1 == time - initialpoint+1:
                    print(addr1)
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_label_'+t1+'.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_'+t1+'.nii')
                else:
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_'+t1+'.nii')
                    Fullsize_regression_1 = niftiread(addr1+'Weights_'+t1+'.nii')

                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1,Fullsize_regression_2, t2, i1, spatial_extend_matrix, addr2, padding)
                dashline()

        else:
            for i1 in range(1, trackbackT+1):
                Fullsize_2 = niftiread(addr2+'Fullsize_label_' + t2+'.nii')
                Fullsize_regression_2 = niftiread(addr2+'Weights_'+t2+'.nii')
                Fullsize_1 = niftiread(addr1+'Fullsize_2_aftertracking_'+t1+'.nii')
                Fullsize_regression_1 = niftiread(addr1+'Weights_'+t1+'.nii')
                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1,Fullsize_regression_2, t2, i1, spatial_extend_matrix, addr2, padding)
                dashline()

        # del Fullsize_1, Fullsize_regression_1,Fullsize_2, Fullsize_regression_2#, Fullsize_1_padding, Fullsize_2_padding, \
        # Fullsize_regression_1_padding, Fullsize_regression_2_padding, Fullsize_1_label, Fullsize_2_label

        # plot tracking
        t1 = str(time)
        t2 = str(time+1)

        # read the correlation calculation results
        correlation_map_padding_show1 = niftiread(folder + t2 + '/' + 'correlation_map_padding_show_traceback1_' + t2 + '.nii')
        correlation_map_padding_hide1 = niftiread(folder + t2 + '/' + 'correlation_map_padding_hide_traceback1_' + t2 + '.nii')

        if time-initialpoint < trackbackT and time > initialpoint:
            for i1 in range(1, time-initialpoint+1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(folder+t2+'/'+'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1)+'_'+t2+'.nii')
                for i2 in range(1, I3dw[0]+padding[0]*2 + 1):  # +1 because python
                    for i3 in range(1, I3dw[1]+padding[1]*2 + 1):
                        for i4 in range(1, I3dw[2]+padding[2]*2 + 1):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[i2,i3,i4] and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2,i3,i4]
        elif time-initialpoint >= trackbackT and time > initialpoint:
            for i1 in range(2, trackbackT+1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_'+ t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(folder + t2 + '/' + 'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 +'.nii')
                correlation_map_padding_hide1_2 = niftiread(folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1)+'_'+t2+'.nii')
                for i2 in range(1, I3dw[0]+padding[0]*2 + 1):  # +1 because python
                    for i3 in range(1, I3dw[1]+padding[1]*2 + 1):
                        for i4 in range(1, I3dw[2]+padding[2]*2 + 1):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[i2, i3, i4] and correlation_map_padding_show1_2[i2, i3, i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2, i3, i4]
        else:
            Registration1 = niftiread(folder + t1 + '/' + 'Registration_' + t1 + '.nii')

        # Read segmentation
        Fullsize_2 = niftiread(folder + t2 + '/Fullsize_' + t2 + '.nii').asType(bool)
        Fullsize_2_2 = np.zeros(shape=(np.shape(Fullsize_2)))

        # crop the expanded sample to its original size
        correlation_map_padding_show2 = correlation_map_padding_show1[21:-1*padding[0], 21:-1*padding[1], 3:-1*padding[2]]
        Fullsize_2_mark = correlation_map_padding_show2

        if time > initialpoint:
            correlation_map_padding_show2_2 = correlation_map_padding_show1_2[21:-1*padding[0], 21:-1*padding[1], 3:-1*padding[2]]
            Fullsize_1 = correlation_map_padding_show2_2
            Fullsize_1[Fullsize_1 == 0] = np.nan
            #if not initial time point, read the fusion data of last time point

            # detector_fusion_old=load(strcat('D:\NEW\',folder,'\',t1,'\fusion_tracking_',t1,'.mat'),'detector3_fusion');
            # for i1=2:2:size(detector_fusion_old.detector3_fusion,1)
            #     detector_fusion_old.detector3_fusion(i1,:)=0;
            # end

            # ------------------------------------------------------------------
            detector_fusion_old = scio.loadmat(folder+t1+'/'+'fusion_tracking_'+t1+'.mat', 'detector3_fusion')
            for i1 in range(2,np.size(detector_fusion_old.detector3_fusion, axis=0)+1,2):
                detector_fusion_old.detector3_fusion[i1,:] = 0

            Fullsize_2_mark[Fullsize_2 == 0] = 0
            # ------------------------------------------------------------------

        del correlation_map_padding_show1, correlation_map_padding_show1_2, correlation_map_padding_hide1, correlation_map_padding_hide1_2

        # Get the object characteristics
        Fullsize_2_mark_BW = Fullsize_2_mark
        Fullsize_2_mark_BW[Fullsize_2_mark_BW > 0] = 1
        Fullsize_2_mark_BW = Fullsize_2_mark_BW.astype(bool)
        Fullsize_2_mark_label, orgnum = measure.label(Fullsize_2_mark, connectivity=1, return_num=True)

        # stats1 = regionprops3(Fullsize_2,'BoundingBox','VoxelList','ConvexHull','Centroid');
        stats1 = pd.DataFrame(measure.regionprops_table(Fullsize_2, properties=('label', 'bbox', 'coords', 'centroid')))
        VoxelList = stats1.coords

        #  sort the objects in descending order of size

        count = np.zeros(stats1.shape[0])
        for i in range(stats1.shape[0]):
            count[i] = np.size(stats1.coords[i], axis=0)
        stats1['Count'] = count.astype(int)

        stats2 = stats1.sort_values(by='Count', axis=0, ascending=False, ignore_index=False)

        detector_fusion = []
        detector_split = []
        detector2_fusion = []
        detector2_split = []

        # stack_after_label[Fullsize_2_mark > 0] = 0  Not used, not initialized

        Fullsize_2_mark[Fullsize_2_mark == 0] = 0

        # Initialize new Registration variables
        newc = 0
        l = np.size(Registration1, axis=0)
        Registration2 = np.zeros(1, 4)
        detector_old = []
        detector_new = []
        detector_numbering = []
        c1 = 1
        c2 = 1
        c3 = 1
        c_numbering = 0
        cc = []

        # Tracking each object
        for i in range(stats2.shape[0]):
            max_object_intensity1 = 0
            max_object_intensity2 = 0
            b = stats2.coords[i]
            if time+1 < 10:
                ttag = '00'
            elif time+1 < 100:
                ttag = '0'
            else:
                ttag = ''

            threeDimg1 = niftiread('/home/nirvan/Desktop/Projects/EcadMyo_08_all/3DImage/' + 'EcadMyo_08/' + 'Ecad/' + 'threeDimg_' +
                                   ttag + str(time+1) + '.nii')
            threeDimg2 = niftiread('/home/nirvan/Desktop/Projects/EcadMyo_08_all/3DImage/' + 'EcadMyo_08/' + 'Myo/' + 'threeDimg_' +
                                   ttag + str(time+1) + '.nii')
            threeDimgPixelList1 = []
            threeDimgPixelList2 = []

            for i1 in range(np.size(b,axis=0)):
                threeDimgPixelList1.append(threeDimg1[b[i1, 0], b[i1, 1], b[i1, 2]])
                threeDimgPixelList2.append(threeDimg2[b[i1, 0], b[i1, 1], b[i1, 2]])

                if threeDimg1[b[i1,0],b[i1,1],b[i1,2]] > max_object_intensity1:
                    max_object_intensity1 = threeDimg1[b[i1,0],b[i1,1],b[i1,2]]

                if threeDimg2[b[i1,0],b[i1,1],b[i1,2]] > max_object_intensity1:
                    max_object_intensity2 = threeDimg2[b[i1,0],b[i1,1],b[i1,2]]

            threeDimgPixelList1 = sorted(threeDimgPixelList1, reverse=True)
            threeDimgPixelList2 = sorted(threeDimgPixelList2, reverse=True)

            # Average the pixels to get average object intensity
            average_object_intensity1 = sum(threeDimgPixelList1)/np.size(threeDimgPixelList1)
            average_object_intensity2 = sum(threeDimgPixelList2)/np.size(threeDimgPixelList2)

            a = []
            a_t_1 = []
            k = boundary(b)
            for i1 in range(np.size(b,axis=0)):
                a.append(Fullsize_2_mark[b[i1,0],b[i1,1],b[i1,2]])
            value = statistics.mode(np.array(a).flatten())
            u, c = np.unique(np.array(a), return_counts=True)
            Value_f = dict(zip(u, c))[value]

            countnan = a.count(np.nan)
            if countnan > Value_f:
                value = np.nan

            if time > startpoint:
                for i1 in range(np.size(b,axis=0)):
                    a_t_1.append(Fullsize_1[b[i1,0],b[i1,1],b[i1,2]])

                value_t_1 = statistics.mode(np.array(a_t_1).flatten())
                u, c = np.unique(np.array(a_t_1), return_counts=True)
                Value_f_t_1 = dict(zip(u, c))[value_t_1]
            
            # Check whether the object has already merged in the last time point
            if not np.isnan(value_t_1) and isempty(intersect(value_t_1, Registration1[:,0])) \
                    and not isempty(intersect(value_t_1, detector_fusion_old.detector3_fusion)): # merge happened in last time point
                detector_numbering.append([value, value_t_1])
                value = value_t_1
                print(value)
            # Check whether the object has alreadybeen tracked in the current time point
            if not isempty(intersect(value,Registration2[:,0])):
                value2 = setdiff(a, Registration2[:,0])

                if not isempty(value2) and np.size(value2,axis=1) > 0 and not isempty(intersect(value2,Registration1[:,0])):
                    value = value2[1, math.ceil(rand()*np.size(value2,axis=1))]

            # If the representatiove of an object is NaN, it means that it is a new object, Assign new ID to it

            if np.isnan(value):
                color = []
                newc += 1

                Registration2[1+n]










    workbook.close()
