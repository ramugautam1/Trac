import numpy as np
import glob as glob
import cv2
import os
from functions import niftiread, niftiwriteF
from PIL import Image
from skimage import io

def czi2nii(folder):
    allfiles = sorted(glob.glob(folder + '/' + '*.czi'))
    for file_i in allfiles:
        savePath = folder + '/nii_files'
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        # im = cv2.imread(file_i)
        im =Image.open(file_i)
        im=np.array(im)
        file_i_name = savePath+'/' + file_i.split('/')[-1].split('.')[0]+'.nii'
        niftiwriteF(im, file_i_name)
        orgname = file_i.split('/')[-1].split('.')[0]

def tif2nii(folder):
    allfiles = sorted(glob.glob(folder + '/' + '*.tif'))
    print(allfiles)
    for file_i in allfiles:
        orgname = file_i.split('/')[-1].split('.')[0]
        savePath = folder + '/nii_files'
        savePath2 = folder + '/nii_files_gt'
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        # im = Image.open(file_i)
        # im.show()
        # im = np.array(im)
        im = io.imread(file_i)
        shape = np.shape(im)
        shapeS = sorted(np.shape(im))

        if shape[-1] > shape[-2] and shape[-4] > shape[-3]:
           im= np.transpose(im,(-1,-2,-4,-3))

        newArr = np.zeros(shape=(512,320,15))
        newArr2 = np.zeros(shape=(512,320,15))
        # newArr[:,:,1:14,:]=im[:,20:300,:]
        if np.size(im,2)==13:
            newArr[:,:,1:14]=im[:,:,:,0]
            newArr2[:,:,1:14]=im[:,:,:,1]

        # #To save as jpg
        # for ir in range(15):
        #     ext = str(0) if ir < 10 else ''
        #     # image = newArr[:,:,ir,0]
        #     print('/home/nirvan/Desktop/newData/NewData/train/' + 'image_' + ext + str(ir) + '.jpeg')
        #     Image.fromarray(newArr[:,:,ir,0]).convert('RGB').save('/home/nirvan/Desktop/newData/NewData/valid/'+ orgname+'_' + ext + str(ir)+ '.jpg')
        #     Image.fromarray(newArr[:, :, ir, 1]).convert('RGB').save('/home/nirvan/Desktop/newData/NewData/valid_labels/' + orgname + 'GT_' + ext + str(ir) + '.jpg')

        print(np.shape(im))
        print(shapeS)
        file_i_name = savePath+'/' + file_i.split('/')[-1].split('.')[0]+'.nii'
        file_i_name2 = savePath2 + '/' + file_i.split('/')[-1].split('.')[0]+'_GT.nii'
        niftiwriteF(newArr, file_i_name)
        niftiwriteF(newArr2,file_i_name2)

    allfiles2 = sorted(glob.glob(folder + '/' + '*.tiff'))
    for file_i in allfiles2:
        savePath = folder + '/nii_files'
        if not os.path.isdir(savePath):
            os.makedirs(savePath)
        im = cv2.imread(file_i)
        file_i_name = savePath + '/' + file_i.split('/')[-1].split('.')[0] + '.nii'
        niftiwriteF(im, file_i_name)
