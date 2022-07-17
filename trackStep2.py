import scipy.io as scio
import numpy as np
from functions import line
from datetime import datetime
import xlsxwriter
import os
import math
import time
import glob as glob
import nibabel as nib

from correlation20220708 import correlation
# from testCorr import correlation
from functions import dashline, starline, niftiread, niftiwrite

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

    filename = folder + 'TrackingID' + str(time) + '.xlsx' # the excel file name to write the tracking result
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
    endpoint = 1  # the time point to stop tracking

    spatial_extend_matrix = np.full((10, 10, 3, depth), 0)  #the weight decay of 'extended search' (not used right now in correlation calculation)

    for i1 in range(0,10):
        for i2 in range(0,10):
            for i3 in range(0,3):
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

                correlation(Fullsize_1,Fullsize_2,Fullsize_regression_1,Fullsize_regression_2,t2,i1,spatial_extend_matrix,addr2,padding)
                starline()


        else:
            for i1 in range(1,trackbackT):
                Fullsize_2 = niftiread(addr2+'Fullsize_label_'+ t2+'.nii')
                Fullsize_regression_2 = niftiread(addr2+'Weights_'+t2+'.nii')
                Fullsize_1 = niftiread(addr1+'Fullsize_2_aftertracking_'+t1+'.nii')
                Fullsize_regression_1 = niftiread(addr1+'Weights_'+t1+'.nii')
                correlation(Fullsize_1,Fullsize_2,Fullsize_regression_1,Fullsize_regression_2,t2,i1,spatial_extend_matrix,addr2, padding)
                dashline()

        # del Fullsize_1, Fullsize_regression_1,Fullsize_2, Fullsize_regression_2#, Fullsize_1_padding, Fullsize_2_padding, \
           # Fullsize_regression_1_padding, Fullsize_regression_2_padding, Fullsize_1_label, Fullsize_2_label



    workbook.close()




