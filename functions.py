import numpy as np
import nibabel as nib
import pandas as pd


def dashline():
    print('---------------------------------------------------------')


def starline():
    print('**********************************************************')


def niftiread(arg):
    return np.asarray(nib.load(arg).dataobj).astype(np.float32).squeeze()


def niftiwrite(a,b):
    nib.save(nib.Nifti1Image(np.uint32(a),affine=np.eye(4)),b)


def line(a):
    return a*50


def rand():
    return np.random.rand()


def size3(arg):
    return [np.size(arg, axis=0), np.size(arg, axis=1), np.size(arg, axis=2)]


