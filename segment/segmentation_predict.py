from __future__ import print_function

from segment.load_data_nii import loadDataGeneral as loadDataGeneral_nii
from segment.load_data_nii import loadDataGeneral_test as loadDataGeneral_test_nii
from segment.load_data import loadDataGeneral
from segment.build_model import build_mymodel
import segment.utils as utils
import segment.FCN as FCN
# import FCN2, models.MobileUNet3D, models.resnet_v23D, models.Encoder_Decoder3D, models.Encoder_Decoder3D_contrib, models.DeepLabp3D, models.DeepLabV33D
# import models.FRRN3D, models.FCN3D, models.GCN3D, models.AdapNet3D, models.ICNet3D, models.PSPNet3D, models.RefineNet3D, models.BiSeNet3D, models.DDSC3D
# import models.DenseASPP3D, models.DeepLabV3_plus3D

from functions import niftiread, niftiwriteF

import matplotlib.pyplot as plt
import numpy as np # Path to csv-file. File should contain X-ray filenames as first column,
        # mask filenames as second column.
import nibabel as nib
# from keras.models import load_model
import math as math
from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure


from skimage.color import hsv2rgb, rgb2hsv, gray2rgb
from skimage import io, exposure
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os,time,cv2, sys, math
import numpy as np
import cv2, glob
import time, datetime
import argparse
import random
import os, sys
import subprocess
import segment.helpers as helpers
import pandas as pd
import csv


def IoU(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    print(y_true.shape, y_pred.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)


def Dice(y_true, y_pred):
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator


def balanced_cross_entropy(beta, y_true, y_pred):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)


def focal_loss(y_true, logits, alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                    weight_a + weight_b) + logits * weight_b

    y_pred = tf.math.sigmoid(logits)
    loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

    return tf.reduce_mean(loss)


def saggital(img):
    """Extracts midle layer in saggital axis and rotates it appropriately."""
    return img[:,  int(img.shape[1] / 2), ::-1].T


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def predictionSampleGeneration(image,startpoint,endpoint):
    sampleAddress = os.path.dirname(image) + '/' + os.path.basename(image).split('.')[0] + '_PredSamples'
    if not os.path.isdir(sampleAddress):
        os.mkdir(sampleAddress)

    I3d = [32, 35, 15]
    I3d2 = [32, 32, 15]
    t1 = startpoint
    t2 = endpoint

    V_sample = niftiread(image)
    for t in range(t1 - 1, t2):
        c_all = 1
        V_sample_t = np.squeeze(V_sample[:, :, :, t, 0])
        if np.mean(V_sample_t) < 0:
            V_sample_t = V_sample_t + 32768
        if not os.path.isdir(sampleAddress + '/' + str(t + 1) + '/'):  # Create directory for output for every timepoint
            os.makedirs(sampleAddress + '/' + str(t + 1) + '/')

        filename1 = sampleAddress + '/' + str(t+1) + '/' + 'idx_pred.csv'
        V_s = np.zeros(shape=(32, 32, 15))
        V_o = np.zeros(shape=(32, 35, 15))

        with open(filename1, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'pathmsk'])
            file.close()

        for i1 in range(0, np.size(V_sample_t, 0), I3d[0]):
            for i2 in range(0, np.size(V_sample_t, 1), I3d[1]):
                a = i1
                b = i1 + I3d[0]
                c = i2
                d = i2 + I3d[1]

                for ix in range(I3d[2]):
                    V_s[:, :, ix] = cv2.resize(V_sample_t[a:b, c:d, ix], (I3d2[0], I3d2[1]),
                                               interpolation=cv2.INTER_LINEAR)
                    # V_s = (V_s / 32768 + 1) / 2

                for ix in range(I3d[2]):
                    V_o[:, :, ix] = V_sample_t[a:b, c:d, ix]

                if c_all < 10:
                    c_all_n = '00' + str(c_all)
                elif c_all < 100:
                    c_all_n = '0' + str(c_all)
                else:
                    c_all_n = str(c_all)
                filename2 = sampleAddress + '/' + str(t + 1) + '/' + 'predimg_' + os.path.basename(image).split('.')[0] + '_' + c_all_n + '.nii'

                with open(filename1, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([os.path.basename(filename2), os.path.basename(filename2)])
                    file.close()

                niftiwriteF(V_s, filename2)
                c_all += 1

    print('Data Preparation Complete!')
    return sampleAddress

def predict(model,image, startpoint, endpoint, op_folder):
    model = model
    image = image
    startpoint = startpoint
    endpoint = endpoint
    op_folder = op_folder
    mode = 'predict'
    
    pdataset = predictionSampleGeneration(image,startpoint,endpoint)
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_balancing', type=str2bool, default=True, help='Whether to use median frequency class weights to balance the classes in the loss')
    parser.add_argument('--continue_training', type=str2bool, default=True, help='Whether to continue training from a checkpoint')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
    
    parser.add_argument('--class_weight_reference', type=str, default="reference/Ecad2020", help='reference you are using.')
    """
    Currently, FC-DenseNet is the best model.
    PSPNet must take input size 192 for 3D
    """
    """
    Try to accommodate the input size.
    Input size for Ecad2017: 128x128x13
    Input size for Ecad2020: 32x32x15
    Input size for Aju2020: 32x32x15
    
    Output size for Ecad2017: 32x35x13
    Output size for Ecad2020: 35x32x15
    Output size for Aju2020: 35x32x15
    """
    args = parser.parse_args()

    img_size = 32

    num_classes = 2

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess=tf.Session(config=config)

    net_input = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 15, 1])
    net_output = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 15, num_classes])

    network = None
    init_fn = None
    print(model)
    if model == "mymodel":
        network = build_mymodel(net_input)
    elif model == "FC-DenseNet":
        network = FCN.build_fc_densenet(net_input)
    '''
    # elif model == "MobileUNet3D-Skip":
    #     network = models.MobileUNet3D.build_mobile_unet3D(net_input, 'MobileUNet3D-Skip', 2)
    # elif model == "FC-DenseNet103":
    #     network = FCN2.build_fc_densenet(net_input,num_classes=num_classes)
    # elif model == "ResNet-101":
    #     network = models.resnet_v23D.resnet_v2_101(net_input, num_classes=num_classes)
    # elif model == "Encoder_Decoder3D":
    #     network = models.Encoder_Decoder3D.build_encoder_decoder(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif model == "Encoder_Decoder3D_contrib":
    #     network = models.Encoder_Decoder3D_contrib.build_encoder_decoder(net_input, num_classes=num_classes)
    # elif model == "DeepLabV3p3D":
    #     network = models.DeepLabp3D.Deeplabv3(net_input, num_classes)
    # elif model == "DeepLabV33D-Res50" or model == "DeepLabV33D-Res101" or model == "DeepLabV33D-Res152":
    #     network = models.DeepLabV33D.build_deeplabv3(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DeepLabV3_plus-Res50" or model == "DeepLabV3_plus-Res101" or model == "DeepLabV3_plus-Res152":
    #     network = models.DeepLabV3_plus3D. build_deeplabv3_plus(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "FRRN-A" or model == "FRRN-B":
    #     network = models.FRRN3D.build_frrn(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "FCN8":
    #     network = models.FCN3D.build_fcn8(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif model == "GCN-Res50" or model == "GCN-Res101" or model == "GCN-Res152":
    #     network = models.GCN3D.build_gcn(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "AdapNet3D":
    #     network = models.AdapNet3D.build_adaptnet(net_input, num_classes=num_classes)
    # elif model == "ICNet-Res50" or model == "ICNet-Res101" or model == "ICNet-Res152":
    #     network = models.ICNet3D.build_icnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=model)
    # elif model == "PSPNet-Res50" or model == "PSPNet-Res101" or model == "PSPNet-Res152":
    #     network = models.PSPNet3D.build_pspnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=model)
    # elif model == "RefineNet-Res50" or model == "RefineNet-Res101" or model == "RefineNet-Res152":
    #     network = models.RefineNet3D.build_refinenet(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "BiSeNet-ResNet50" or model == "BiSeNet-Res101" or model == "BiSeNet-Res152":
    #     network = models.BiSeNet3D.build_bisenet(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DDSC-ResNet50" or model == "DDSC-Res101" or model == "DDSC-Res152":
    #     network = models.DDSC3D.build_ddsc(net_input, num_classes=num_classes, preset_model=model)
    # elif model == "DenseASPP-ResNet50" or model == "DenseASPP-Res101" or model == "DenseASPP-Res152":
    #     network = models.DenseASPP3D.build_dense_aspp(net_input, num_classes=num_classes, preset_model=model)
    '''

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    if init_fn is not None:
        init_fn(sess)

    model_checkpoint_name = "checkpoints/" + model + '/' + 'Data_Ecad2020' + "/latest_model_" + "_" + 'Data_Ecad2020' + ".ckpt"

    if args.continue_training or not mode == "train":
        print('Loaded latest model checkpoint')
        print(model_checkpoint_name)
        saver.restore(sess, model_checkpoint_name)
#---------------------------------------------------------------------------------------------------------------------------------------------

    for tt in range(startpoint, endpoint+1):
        print("\n***** Begin prediction *****")
        print("Dataset -->", pdataset.split('/')[-1])
        print("Model -->", model)
        print("Num Classes -->", num_classes)
        print("time point -->", tt)
        print("")

        # Create directories if needed
        addrSegRes = op_folder

        if not os.path.isdir("%s/%s/%s/%s" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt)):
            os.makedirs("%s/%s/%s/%s" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt))
        print(os.path.isdir("%s/%s/%s/%s" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt)))

        csv_path_val = pdataset + '/'+str(tt)+'/idx_pred.csv'
        # Path to the folder with images. Images will be read from path + path_from_csv


        path2 = csv_path_val[:csv_path_val.rfind('/')] + '/'  # + str(tt) + '/'

        df = pd.read_csv(csv_path_val)

        # Load test data
        input_image_pred, gt= loadDataGeneral(df, path2, img_size)
        print(input_image_pred[0].shape,gt[0].shape)

        # Run testing on ALL test images
        for ind in range(len(input_image_pred)):
            input_image = np.expand_dims(
                np.float32(input_image_pred[ind]), axis=0) # / 255.0
            print(input_image.shape)

            sys.stdout.write("\rRunning predict image %d / %d" % (ind + 1, len(input_image_pred)))
            sys.stdout.flush()

            st = time.time()
            #output_image = sess.run(network, feed_dict={net_input: input_image})
            output_image, stack = sess.run(network, feed_dict={net_input: input_image})

            #model_checkpoint_name = "checkpoints/" + model + "/latest_model_" + "_" + args.dataset + ".ckpt"
            #new_saver = tf.train.import_meta_graph('/home/scw4750/Liuhongkun/tfrecord/zooscan/Alexnet/Modal/model20170226041552612/mymodel.meta')

            #run_times_list.append(time.time() - st)


            output_image = np.array(output_image[0, :, :, :])
            input_image = np.array(input_image[0, :, :, :])
            stack = np.array(stack[0, :, :, :, :])

            # try to accommodate the size
            w, l, h, c = 35, 32, 15, 64

            #output_image_resize = np.zeros((w, l, h))
            stack_resize = np.zeros((l, w, h, c))
            for idx in range(stack.shape[2]):
                #img = output_image[:, :, idx, 1]
                #img_sm = cv2.resize(img, (w, l), interpolation=cv2.INTER_LINEAR)
                #output_image_resize[:, :, idx] = img_sm
                for idx2 in range(c):
                    stk = stack[:, :, idx, idx2]
                    stk_sm = cv2.resize(stk, (w, l), interpolation=cv2.INTER_LINEAR)
                    stack_resize[:, :, idx, idx2] = stk_sm



            new_image = nib.Nifti1Image(output_image, affine=np.eye(4))
            input_image = nib.Nifti1Image(input_image, affine=np.eye(4))
            final_weights_output = nib.Nifti1Image(stack_resize, affine=np.eye(4))
            if ind<10:
                nib.save(new_image, "%s/%s/%s/%s/Z00%s_regression.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(input_image, "%s/%s/%s/%s/Z00%s_input.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(final_weights_output, "%s/%s/%s/%s/Z00%s_final_weights.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
            elif ind<100:
                nib.save(new_image, "%s/%s/%s/%s/Z0%s_regression.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(input_image, "%s/%s/%s/%s/Z0%s_input.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(final_weights_output, "%s/%s/%s/%s/Z0%s_final_weights.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
            else:
                nib.save(new_image, "%s/%s/%s/%s/Z%s_regression.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(input_image, "%s/%s/%s/%s/Z%s_input.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
                nib.save(final_weights_output, "%s/%s/%s/%s/Z%s_final_weights.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))

            output_image_resize = helpers.reverse_one_hot(output_image)
            #out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            new_image = nib.Nifti1Image(output_image_resize, affine=np.eye(4))
            if ind < 10:
                nib.save(new_image, "%s/%s/%s/%s/Z00%s_class.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
            elif ind < 100:
                nib.save(new_image, "%s/%s/%s/%s/Z0%s_class.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))
            else:
                nib.save(new_image, "%s/%s/%s/%s/Z%s_class.nii" % (addrSegRes, os.path.basename(image).split('.')[0]+'_SegmentationOutput', model, tt, str(ind)))


