# # import pandas as pd
# # import math
# # import numpy as np
# # import matplotlib as mpl
# # import matplotlib.pyplot as plt
# # import os
# #
# #
# # def intersect(a, b):
# #     a1, ia = np.unique(a, return_index=True)
# #     b1, ib = np.unique(b, return_index=True)
# #     aux = np.concatenate((a1, b1))
# #     aux.sort()
# #     c = aux[:-1][aux[1:] == aux[:-1]]
# #     return c
# #
# #
# # saveFolder = '/home/nirvan/Desktop/AppTestRun/FamilyTrees'
# # # saveFolder = ftFolder
# # if not os.path.exists(saveFolder):
# #     os.makedirs(saveFolder)
# #
# # # df = pd.read_excel('/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/TrackingID2022-08-02 18:11:23.905684.xlsx', sheet_name='Sheet1')
# # df = pd.read_excel('/home/nirvan/Desktop/AppTestRun/TrackResult/TrackingID2022-11-23 10:42:17.314780.xlsx', sheet_name='Sheet1')
# # # df = pd.read_excel(excelFile, sheet_name='Sheet1')
# # # print(df.head())
# # lst = []
# # # print(df.iloc[0:10,0:10])
# #
# # for ix in range(0, df.shape[0]):
# #     if (not pd.isna(df.iloc[ix, 0]) and df.iloc[ix, 0] == ix + 2) or (df.iloc[ix, :] == 'new').any():
# #         lst.append(ix + 2)
# #
# # # print(lst)
# # # print('\n\n')
# #
# # lst2 = []
# #
# # x = df.stack().value_counts()
# # x.pop('new')
# #
# # keys = np.asarray(x.keys())[0:].astype(np.int32)
# # vals = np.asarray(x.values)[0:].astype(np.int32)
# #
# # for ill in range(100):
# #     print(keys[ill],vals[ill])
# #
# #
# # for i, v in enumerate(vals):
# #     if v > 10:
# #         lst2.append(keys[i])
# #
# # lst2.sort()
# #
# # targetIds = intersect(lst, lst2)
# # print('\n')
# # print(np.size(targetIds), 'family trees.')
# # # print(list(targetIds))
# # ftlst = []
# #
# # ftnum = 0
# #
# # print('\n=================================================\n')
# #
# # for tid in targetIds:
# #
# #     set = []
# #     indexlist = []
# #     print("target id: " + str(tid), end=' ')
# #     for ix in range(0, df.shape[0]):
# #         for jx in range(0, df.shape[1], 2):
# #             if df.iloc[ix, jx] == tid and ix + 2 not in indexlist:
# #                 indexlist.append(ix + 2)
# #                 break
# #     # print("indexlist:  ",indexlist)
# #
# #     df2 = df.copy()
# #     k2 = str(int(df.shape[1] / 2 + 1)) + '.1'
# #     k1 = str(int(df.shape[1] / 2 + 1))
# #     df2[k2] = df.loc[:, k1]
# #     # print(df2.head())
# #
# #     timelist = []
# #     for inndx, idx in enumerate(indexlist):
# #         for ix in range(idx - 2, df2.shape[0]):
# #             size1 = np.size(timelist)
# #             for jx in range(0, df2.shape[1], 2):
# #                 if jx == 0 and df2.iloc[ix, jx] == idx:
# #                     timelist.append(1)
# #                     break
# #
# #                 elif jx > 0 and jx < (df2.shape[1] - 1):
# #                     if df2.iloc[ix, jx - 1] == idx:
# #                         timelist.append(1 + (jx / 2))
# #                         break
# #
# #                 elif jx == df2.shape[1] - 1 and df2.iloc[ix, df2.shape[1] - 1] == idx:
# #                     timelist.append(1 + jx / 2)
# #                     break
# #
# #             # print(idx,df2.iloc[ix,jx-1],df2.iloc[ix,jx-1]==idx, df2.iloc[ix,0]==idx,1+jx/2)
# #             size2 = np.size(timelist)
# #             # print(timelist, size1, size2)
# #             if (size2 > size1):
# #                 break
# #     timelist = [int(tl) for tl in timelist[0:len(indexlist)]]
# #     # print('timelist:  ', timelist)
# #
# #     timeendlist = []
# #     for idx in indexlist:
# #         for ix in range(idx - 2, df.shape[0]):
# #             size3 = np.size(timeendlist)
# #             for jx in range(df.shape[1] - 1, 0, -2):
# #                 if df.iloc[ix, jx] == idx or df.iloc[ix, jx - 1] == idx:
# #                     timeendlist.append(1 + math.ceil(jx / 2))
# #                     break
# #             size4 = np.size(timeendlist)
# #             if (size4 > size3):
# #                 break
# #     timelist = [int(tl) for tl in timelist]
# #     # print('timeendlist:  ', timeendlist)
# #
# #     parentlist = []
# #     for index, idx in enumerate(indexlist):
# #         if index == 0:
# #             parent = idx
# #         else:
# #             parent = df.iloc[idx - 2, 2 * (timelist[index] - 2)]
# #         parentlist.append(parent)
# #
# #     print(u'\u2713') if len(indexlist) == len(timelist) == len(timeendlist) == len(parentlist) else print('error')
# #     set.append(indexlist)
# #     set.append(timelist)
# #     set.append(timeendlist)
# #     set.append(parentlist)
# #
# #     ftlst.append(set)
# #
# # print('\n=================================================\n')
# # # for a in range(0,len(ftlst)):
# # # 	for b in range(0,len(ftlst[a])):
# # # 		print(ftlst[a][b])
# # # 	print('')
# #
# # # print('\n=================================================\n')
# # print(ftlst)
# # print('\n=================================================\n')
# #
# # colors = ['#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF',
# #           '#00C957', '#8B6914',
# #           '#FF1493', '#8FBC8F', '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103',
# #           '#458B00', '#FFB90F',
# #           '#E06E00', '#B23EEE', '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878',
# #           '#FE7256', '#EE3B3B',
# #           '#8EE5EE', '#FF6103', '#458B00', '#FFB90F', '#006400', '#B23AEE', '#00BFFF', '#00C957', '#8B6914',
# #           '#FF1493', '#8FBC8F',
# #           '#CD661D', '#8B8878', '#FF7256', '#0000CD', '#EE3B3B', '#8EE5EE', '#FF6103', '#458B00', '#FFB90F',
# #           '#E06E00', '#B23EEE',
# #           '#E0BFFF', '#0EC957', '#8E6914', '#FA1493', '#EFBC8F', '#CE661D', '#8E8878', '#FE7256', '#8E6914',
# #           '#FA1493', '#EFBC8F']
# #
# # for index, ft in enumerate(ftlst):
# #     fig = plt.figure(figsize=(52, 27))
# #     ax = plt.subplot()
# #     ax.set_xlim(0, max(ft[2]) + 5)
# #     ax.set_ylim(0, len(ft[0]) + 1)
# #     ax.set_xlabel('Time Points (t)', fontsize=17, color='k')
# #     mpl.rc('xtick', labelsize=17)
# #     mpl.rc('ytick', labelsize=17)
# #     plt.xticks(rotation=0)
# #     plt.yticks(color='w')
# #     k = 1
# #
# #     notplottedlist = []
# #     for ind, itm in enumerate(ft[0]):
# #         if ind != 0 and (itm not in ft[3]) and ((ft[2][ind] - ft[1][ind]) < 4):
# #             notplottedlist.append(ind)
# #     print(notplottedlist)
# #
# #     for i, j in enumerate((ft[0])):
# #         if i not in notplottedlist:  # min time filter
# #             for k in range(ft[1][i] - 1, ft[2][i]):
# #                 ax.scatter(k + 1, i + 1, c=colors[i], s=400)
# #                 ax.text(ft[2][i] + 1, i + 1, str(ft[0][i]), fontsize=30 if i == 0 else 20)
# #             plt.plot()
# #             for iii in range(1, len(ft[0])):
# #                 if iii not in notplottedlist:  # min time filter
# #                     l = ft[0].index(ft[3][iii])
# #                     plt.plot([ft[1][iii] - 1, ft[1][iii]], [l + 1, iii + 1], c=colors[iii], linewidth=1)
# #             plt.plot([ft[1][i], ft[2][i]], [i + 1, i + 1], c='k', linewidth=1)
# #
# #     prefix = '00' if ft[0][0] < 10 else '0' if ft[0][0] < 100 else ''
# #     filename = saveFolder + '/' + 'FT_ID_' + prefix + str(ft[0][0]) + '.png'
# #     plt.savefig(filename)
# #     plt.close(fig)
#
# import pandas as pd
# import numpy as np
# indexlist=[1,2,3,4]
# timelist=[11,22,33,44]
# timeendlist=[44,55,66,77]
# parentlist=[4,5,5,9]
# mydf = pd.DataFrame()
# mydf['index'] = indexlist
# mydf['timestart'] = timelist
# mydf['timeend'] = timeendlist
# mydf['parent'] = parentlist
#
# print(mydf)
#
# # 'index', 'timestart', 'timeend', 'parent'
import glob

import numpy as np
import os
import cv2
import csv

from functions import niftiread, niftiwriteF
# # image='/home/nirvan/Desktop/Projects/EcadMyo_08_all/EcadMyo_08.nii' #sample
# image ='/home/nirvan/Desktop/newData/newSample.nii' #sample
# image1='/home/nirvan/Desktop/newData/newSampleT43.nii' #sample
# image2 = '/home/nirvan/Desktop/newData/newSampleT44.nii'
# startpoint = 1
# endpoint = 2
# # Files1 = niftiread(addr1+'EcadMyo_088.nii')
#
# sampleAddress = os.path.dirname(image) + '/' + os.path.basename(image).split('.')[0] + '_PredSamples' + '/'
# if not os.path.isdir(sampleAddress):
#     os.mkdir(sampleAddress)
#
# I3d = [32, 35, 15]
# I3d2 = [32, 32, 15]
# t1 = startpoint
# t2 = endpoint
#
# V_sample1 = niftiread(image1)
# V_sample2 = niftiread(image2)
# V_sample = np.zeros(shape=[512,280,15,2,2])
# V_sample[:,:,:,0,:]=V_sample1
# V_sample[:,:,:,1,:] = V_sample2
#
# niftiwriteF(V_sample,image)
#
# print(np.shape(V_sample))
# for t in range(t1 - 1, t2):
#     c_all = 1
#     V_sample_t = np.squeeze(V_sample[:, :, :, t, 0])
#     if np.mean(V_sample_t) < 0:
#         V_sample_t = V_sample_t + 32768
#     if not os.path.isdir(sampleAddress + str(t + 1) + '/'):  # Create directory for output for every timepoint
#         os.makedirs(sampleAddress + str(t + 1) + '/')
#
#     filename1 = sampleAddress + str(t+1) + '/' + 'idx_pred.csv'
#
#     V_s = np.zeros(shape=(32, 32, 15))
#     V_o = np.zeros(shape=(32, 35, 15))
#
#     with open(filename1, 'w') as file:
#         writer = csv.writer(file)
#         writer.writerow(['path','pathmsk'])
#         file.close()
#
#     for i1 in range(0, np.size(V_sample_t, 0), I3d[0]):
#         for i2 in range(0, np.size(V_sample_t, 1), I3d[1]):
#             a = i1
#             b = i1 + I3d[0]
#             c = i2
#             d = i2 + I3d[1]
#
#             for ix in range(I3d[2]):
#                 V_s[:, :, ix] = cv2.resize(V_sample_t[a:b, c:d, ix], (I3d2[0], I3d2[1]),
#                                            interpolation=cv2.INTER_LINEAR)
#                 # V_s = (V_s / 32768 + 1) / 2
#
#             for ix in range(I3d[2]):
#                 V_o[:, :, ix] = V_sample_t[a:b, c:d, ix]
#
#             if c_all < 10:
#                 c_all_n = '00' + str(c_all)
#             elif c_all < 100:
#                 c_all_n = '0' + str(c_all)
#             else:
#                 c_all_n = str(c_all)
#             filename2 = sampleAddress + str(t + 1) + '/' + 'predimg_' + os.path.basename(image).split('.')[
#                 0] + '_' + c_all_n + '.nii'
#             print(filename2)
#
#             with open(filename1,'a') as file:
#                 writer=csv.writer(file)
#                 writer.writerow([os.path.basename(filename2), os.path.basename(filename2)])
#                 file.close()
#
#             niftiwriteF(V_s, filename2)
#
#
#             c_all += 1
#
# print('Data Preparation Complete!')


### Create Training Data


# I3d = [32, 32, 15]
#
# files = glob.glob('/home/nirvan/Desktop/newData/NewData/train/nii_files/'+'*.nii')
# files2 = glob.glob('/home/nirvan/Desktop/newData/NewData/train/nii_files_gt/'+'*.nii')
# folder1 = '/home/nirvan/Desktop/newData/NewData/train/nii_files'
#
# for t,image in enumerate(files):
#     V_sample = niftiread(image)
#     c_all = 1
#     if np.mean(V_sample) < 0:
#         V_sample = V_sample + 32768
#
#     filename1 = '/home/nirvan/Desktop/newData/NewData/train/nii_files/' + 'idx_train.csv'
#     V_s = np.zeros(shape=(32, 32, 15))
#
#     with open(filename1, 'w') as file:
#         writer = csv.writer(file)
#         writer.writerow(['path','pathmsk'])
#         file.close()
#
#     for i1 in range(0, np.size(V_sample, 0), I3d[0]):
#         for i2 in range(0, np.size(V_sample, 1), I3d[1]):
#             a = i1
#             b = i1 + I3d[0]
#             c = i2
#             d = i2 + I3d[1]
#
#             V_s[:, :,:] = V_sample[a:b, c:d, :]
#
#             if c_all < 10:
#                 c_all_n = '00' + str(c_all)
#             elif c_all < 100:
#                 c_all_n = '0' + str(c_all)
#             else:
#                 c_all_n = str(c_all)
#
#             filename2 = folder1 + '/' + 'trainimg_' + os.path.basename(image).split('.')[
#                 0] + '_' + c_all_n + '.nii'
#             filename2x = folder1 + '/' + 'trainimgGT_' + os.path.basename(image).split('.')[
#                 0] + '_GT_' + c_all_n + '.nii'
#             print(filename2)
#
#             with open(filename1,'a') as file:
#                 writer=csv.writer(file)
#                 writer.writerow([os.path.basename(filename2),os.path.basename(filename2x)])
#                 file.close()
#
#             niftiwriteF(V_s, filename2)
#
#
#             c_all += 1
#
#
# for t,image in enumerate(files2):
#     V_sample = niftiread(image)
#     c_all = 1
#     if np.mean(V_sample) < 0:
#         V_sample = V_sample + 32768
#
#     filename1 = '/home/nirvan/Desktop/newData/NewData/train/nii_files/' + 'idx_train.csv'
#     V_s = np.zeros(shape=(32, 32, 15))
#
#     with open(filename1, 'a') as file:
#         writer = csv.writer(file)
#         writer.writerow(['pathmsk'])
#         file.close()
#
#     for i1 in range(0, np.size(V_sample, 0), I3d[0]):
#         for i2 in range(0, np.size(V_sample, 1), I3d[1]):
#             a = i1
#             b = i1 + I3d[0]
#             c = i2
#             d = i2 + I3d[1]
#
#             V_s[:, :,:] = V_sample[a:b, c:d, :]
#
#             if c_all < 10:
#                 c_all_n = '00' + str(c_all)
#             elif c_all < 100:
#                 c_all_n = '0' + str(c_all)
#             else:
#                 c_all_n = str(c_all)
#
#             filename2 = folder1 + '/' + 'trainimgGT_' + os.path.basename(image).split('.')[
#                 0] + '_' + c_all_n + '.nii'
#             print(filename2)
#
#             # with open(filename1,'a') as file:
#             #     writer=csv.writer(file)
#             #     writer.writerow([os.path.basename(filename2)])
#             #     file.close()
#
#             niftiwriteF(V_s, filename2)
#
#
#             c_all += 1


#### Create Validation data
I3d = [32, 32, 15]

files = glob.glob('/home/nirvan/Desktop/newData/NewData/valid/nii_files/'+'*.nii')
files2 = glob.glob('/home/nirvan/Desktop/newData/NewData/valid/nii_files_gt/'+'*.nii')
folder1 = '/home/nirvan/Desktop/newData/NewData/valid/nii_files'

for t,image in enumerate(files):
    V_sample = niftiread(image)
    c_all = 1
    if np.mean(V_sample) < 0:
        V_sample = V_sample + 32768

    filename1 = '/home/nirvan/Desktop/newData/NewData/valid/nii_files/' + 'idx_valid.csv'
    V_s = np.zeros(shape=(32, 32, 15))

    with open(filename1, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['path','pathmsk'])
        file.close()

    for i1 in range(0, np.size(V_sample, 0), I3d[0]):
        for i2 in range(0, np.size(V_sample, 1), I3d[1]):
            a = i1
            b = i1 + I3d[0]
            c = i2
            d = i2 + I3d[1]

            V_s[:, :,:] = V_sample[a:b, c:d, :]

            if c_all < 10:
                c_all_n = '00' + str(c_all)
            elif c_all < 100:
                c_all_n = '0' + str(c_all)
            else:
                c_all_n = str(c_all)

            filename2 = folder1 + '/' + 'validimg_' + os.path.basename(image).split('.')[
                0] + '_' + c_all_n + '.nii'
            filename2x = folder1 + '/' + 'validimgGT_' + os.path.basename(image).split('.')[
                0] + '_GT_' + c_all_n + '.nii'
            print(filename2)

            with open(filename1,'a') as file:
                writer=csv.writer(file)
                writer.writerow([os.path.basename(filename2),os.path.basename(filename2x)])
                file.close()

            niftiwriteF(V_s, filename2)


            c_all += 1


for t,image in enumerate(files2):
    V_sample = niftiread(image)
    c_all = 1
    if np.mean(V_sample) < 0:
        V_sample = V_sample + 32768

    filename1 = '/home/nirvan/Desktop/newData/NewData/valid/nii_files/' + 'idx_valid.csv'
    V_s = np.zeros(shape=(32, 32, 15))

    with open(filename1, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(['pathmsk'])
        file.close()

    for i1 in range(0, np.size(V_sample, 0), I3d[0]):
        for i2 in range(0, np.size(V_sample, 1), I3d[1]):
            a = i1
            b = i1 + I3d[0]
            c = i2
            d = i2 + I3d[1]

            V_s[:, :,:] = V_sample[a:b, c:d, :]

            if c_all < 10:
                c_all_n = '00' + str(c_all)
            elif c_all < 100:
                c_all_n = '0' + str(c_all)
            else:
                c_all_n = str(c_all)

            filename2 = folder1 + '/' + 'validimgGT_' + os.path.basename(image).split('.')[
                0] + '_' + c_all_n + '.nii'
            print(filename2)

            # with open(filename1,'a') as file:
            #     writer=csv.writer(file)
            #     writer.writerow([os.path.basename(filename2)])
            #     file.close()

            niftiwriteF(V_s, filename2)


            c_all += 1
