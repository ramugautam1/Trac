import scipy.io as scio
import numpy as np
import pandas as pd
from functions import line
from datetime import datetime
import xlsxwriter
import os
import math
import time
import glob as glob
import nibabel as nib
from skimage import measure

from correlation20220708 import correlation
# from testCorr import correlation
from functions import dashline, starline, niftiread, niftiwrite, getVoxelList

def trackStep2():

    starline()  # print **************************************
    print('step 2 start')
    starline()
    colormap = scio.loadmat('/home/nirvan/Desktop/Projects/MATLAB CODES/colormap.mat')
    I3dw = [512, 280, 15]
    padding = [20, 20, 2]
    time = datetime.now()

    folder = '/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08_Tracking_Result/'
    trackbackT = 2

    if not os.path.isdir(folder):
        print(os.makedirs(folder))

    filename = folder + 'TrackingID' + str(time) + '.xlsx'  # the excel file name to write the tracking result
    workbook = xlsxwriter.Workbook(filename)

    dashline()

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

    depth = 64  # the deep features to take in correlation calculation
    initialpoint = 1  # the very first time point of all samples
    startpoint = 1  # the time point to start tracking
    endpoint = 41  # the time point to stop tracking

    spatial_extend_matrix = np.full((10, 10, 3, depth), 0)  #the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0, 10):
        for i2 in range(0, 10):
            for i3 in range(0, 3):
                spatial_extend_matrix[i1, i2, i3, :] = math.exp(((i1+1-5)+(i2+1-5)+(i3+1-2))/20)

    for tim in range(startpoint, endpoint+1):
        tic = time.time()
        print('time point: ' + str(tim))
        t1 = str(tim)
        t2 = str(tim+1)
        worksheet1.write(0, tim * 2 - 2, str(t1))
        worksheet1.write(0, tim * 2 - 1, str(t2))
        worksheet3.write(0, tim * 2 - 1, str(t2))
        worksheet4.write(0, tim * 2 - 1, str(t2))
        worksheet5.write(0, tim * 2 - 1, str(t2))
        worksheet6.write(0, tim * 2 - 1, str(t2))
        worksheet7.write(0, tim * 2 - 1, str(t2))
        worksheet8.write(0, tim * 2 - 1, str(t2))
        worksheet9.write(0, tim * 2 - 1, str(t2))
        worksheet10.write(0, tim * 2 - 1, str(t2))
        worksheet11.write(0, tim * 2 - 1, str(t2))
        worksheet12.write(0, tim * 2 - 1, str(t2))

        addr1 = folder + t1 + '/'
        addr2 = folder + t2 + '/'
        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)
        # if not os.path.isdir(folder):
        #     os.makedirs(addr1)
        Files1 = sorted(glob.glob(addr1 + '*.nii'))
        Files2 = sorted(glob.glob(addr2+'.nii'))

        if tim-initialpoint < trackbackT: #calculating correlation for start time points (e.g. time=2)
            for i1 in range(1, tim-initialpoint+1+1):
                print(f'time  point: {tim}')
                print(addr2)
                Fullsize_2 = niftiread(addr2 + 'Fullsize_label_' + t2 + '.nii')
                Fullsize_regression_2 = niftiread(addr2 + 'Weights_' + t2 + '.nii')
                if i1 == tim - initialpoint+1:
                    print(addr1)
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_label_'+t1+'.nii')
                    Fullsize_regression_1 = niftiread(addr1 + 'Weights_'+t1+'.nii')
                else:
                    Fullsize_1 = niftiread(addr1 + 'Fullsize_2_aftertracking_'+t1+'.nii')
                    Fullsize_regression_1 = niftiread(addr1+'Weights_'+t1+'.nii')

                correlation(Fullsize_1, Fullsize_2, Fullsize_regression_1,Fullsize_regression_2, t2, i1, spatial_extend_matrix, addr2, padding)
                starline()


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


        #plot tracking
        t1 = str(tim)
        t2 = str(tim+1)

        # read the correlation calculation results
        correlation_map_padding_show1 = niftiread(folder + t2 + '/' + 'correlation_map_padding_show_traceback1_' + t2 + '.nii')
        correlation_map_padding_hide1 = niftiread(folder + t2 + '/' + 'correlation_map_padding_hide_traceback1_' + t2 + '.nii')

        if tim-initialpoint < trackbackT and tim > initialpoint:
            for i1 in range(1, tim-initialpoint+1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_' + t1 + '.nii')
                correlation_map_padding_show1_2 = niftiread(folder+t2+'/'+'correlation_map_padding_show_traceback' + str(i1) + '_' + t2 + '.nii')
                correlation_map_padding_hide1_2 = niftiread(folder + t2 + '/' + 'correlation_map_padding_hide_traceback' + str(i1)+'_'+t2+'.nii')
                for i2 in range(1, I3dw[0]+padding[0]*2 + 1):  # +1 because python
                    for i3 in range(1, I3dw[1]+padding[1]*2 + 1):
                        for i4 in range(1, I3dw[2]+padding[2]*2 + 1):
                            if correlation_map_padding_hide1[i2, i3, i4] < correlation_map_padding_hide1_2[i2,i3,i4] and correlation_map_padding_show1_2[i2,i3,i4] != 0:
                                correlation_map_padding_show1[i2, i3, i4] = correlation_map_padding_show1_2[i2,i3,i4]
        elif tim-initialpoint >= trackbackT and tim > initialpoint:
            for i1 in range(2, trackbackT+1):
                Registration1 = niftiread(folder + t1 + '/' + 'Registration2_tracking_'+t1+'.nii')
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

        if tim > initialpoint:
            correlation_map_padding_show2_2 = correlation_map_padding_show1_2[21:-1*padding[0], 21:-1*padding[1], 3:-1*padding[2]]
            Fullsize_1 = correlation_map_padding_show2_2
            Fullsize_1[Fullsize_1 == 0] = np.nan
            #if not initial time point, read the fusion data of last time point

            # # # detector_fusion_old=load(strcat('D:\NEW\',folder,'\',t1,'\fusion_tracking_',t1,'.mat'),'detector3_fusion');
            # # # for i1=2:2:size(detector_fusion_old.detector3_fusion,1)
            # # #     detector_fusion_old.detector3_fusion(i1,:)=0;
            # # # end

            Fullsize_2_mark[Fullsize_2 == 0] = 0

            # ------------------------------------------------------------------






        del correlation_map_padding_show1, correlation_map_padding_show1_2, correlation_map_padding_hide1, correlation_map_padding_hide1_2

        # Get the object characteristics
        Fullsize_2_mark_BW = Fullsize_2_mark
        Fullsize_2_mark_BW[Fullsize_2_mark_BW > 0] = 1
        Fullsize_2_mark_BW = Fullsize_2_mark_BW.astype(bool)
        Fullsize_2_mark_label, orgnum = measure.label(Fullsize_2_mark, connectivity=1, return_num=True)

        # stats1 = regionprops3(Fullsize_2,'BoundingBox','VoxelList','ConvexHull','Centroid');
        stats1 = pd.DataFrame(measure.regionprops_table(Fullsize_2, properties=('label', 'bbox', 'centroid')))
        voxels = getVoxelList(Fullsize_2, orgnum=orgnum)

        stats1 = stats1.join(voxels["VoxelList"])  #can't do that!!!! the objects will not be on the same row














    workbook.close()




