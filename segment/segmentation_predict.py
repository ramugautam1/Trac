from __future__ import print_function

from load_data_nii import loadDataGeneral as loadDataGeneral_nii
from segment.load_data_nii import loadDataGeneral_test as loadDataGeneral_test_nii
from segment.load_data import loadDataGeneral
from segment.build_model import build_mymodel
import segment.utils as utils
import segment.FCN as FCN
# import FCN2, models.MobileUNet3D, models.resnet_v23D, models.Encoder_Decoder3D, models.Encoder_Decoder3D_contrib, models.DeepLabp3D, models.DeepLabV33D
# import models.FRRN3D, models.FCN3D, models.GCN3D, models.AdapNet3D, models.ICNet3D, models.PSPNet3D, models.RefineNet3D, models.BiSeNet3D, models.DDSC3D
# import models.DenseASPP3D, models.DeepLabV3_plus3D

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

def predict(model,image, startpoint, endpoint, op_folder):
    model = model
    image = image
    startpoint = startpoint
    endpoint = endpoint
    op_folder = op_folder

    parser = argparse.ArgumentParser()
    parser.add_argument('--class_balancing', type=str2bool, default=True, help='Whether to use median frequency class weights to balance the classes in the loss')
    parser.add_argument('--continue_training', type=str2bool, default=True, help='Whether to continue training from a checkpoint')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images in each batch')
    parser.add_argument('--pdataset', type=str, default="EcadMyo_08", help='Prediction Dataset you are using.')
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
    print(args.model)
    if args.model == "mymodel":
        network = build_mymodel(net_input)
    elif args.model == "FC-DenseNet":
        network = FCN.build_fc_densenet(net_input)
    '''
    # elif args.model == "MobileUNet3D-Skip":
    #     network = models.MobileUNet3D.build_mobile_unet3D(net_input, 'MobileUNet3D-Skip', 2)
    # elif args.model == "FC-DenseNet103":
    #     network = FCN2.build_fc_densenet(net_input,num_classes=num_classes)
    # elif args.model == "ResNet-101":
    #     network = models.resnet_v23D.resnet_v2_101(net_input, num_classes=num_classes)
    # elif args.model == "Encoder_Decoder3D":
    #     network = models.Encoder_Decoder3D.build_encoder_decoder(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif args.model == "Encoder_Decoder3D_contrib":
    #     network = models.Encoder_Decoder3D_contrib.build_encoder_decoder(net_input, num_classes=num_classes)
    # elif args.model == "DeepLabV3p3D":
    #     network = models.DeepLabp3D.Deeplabv3(net_input, num_classes)
    # elif args.model == "DeepLabV33D-Res50" or args.model == "DeepLabV33D-Res101" or args.model == "DeepLabV33D-Res152":
    #     network = models.DeepLabV33D.build_deeplabv3(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "DeepLabV3_plus-Res50" or args.model == "DeepLabV3_plus-Res101" or args.model == "DeepLabV3_plus-Res152":
    #     network = models.DeepLabV3_plus3D. build_deeplabv3_plus(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "FRRN-A" or args.model == "FRRN-B":
    #     network = models.FRRN3D.build_frrn(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "FCN8":
    #     network = models.FCN3D.build_fcn8(net_input, num_classes=num_classes)
    #     # RefineNet requires pre-trained ResNet weights
    # elif args.model == "GCN-Res50" or args.model == "GCN-Res101" or args.model == "GCN-Res152":
    #     network = models.GCN3D.build_gcn(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "AdapNet3D":
    #     network = models.AdapNet3D.build_adaptnet(net_input, num_classes=num_classes)
    # elif args.model == "ICNet-Res50" or args.model == "ICNet-Res101" or args.model == "ICNet-Res152":
    #     network = models.ICNet3D.build_icnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=args.model)
    # elif args.model == "PSPNet-Res50" or args.model == "PSPNet-Res101" or args.model == "PSPNet-Res152":
    #     network = models.PSPNet3D.build_pspnet(net_input, [img_size, img_size, 13], num_classes=num_classes, preset_model=args.model)
    # elif args.model == "RefineNet-Res50" or args.model == "RefineNet-Res101" or args.model == "RefineNet-Res152":
    #     network = models.RefineNet3D.build_refinenet(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "BiSeNet-ResNet50" or args.model == "BiSeNet-Res101" or args.model == "BiSeNet-Res152":
    #     network = models.BiSeNet3D.build_bisenet(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "DDSC-ResNet50" or args.model == "DDSC-Res101" or args.model == "DDSC-Res152":
    #     network = models.DDSC3D.build_ddsc(net_input, num_classes=num_classes, preset_model=args.model)
    # elif args.model == "DenseASPP-ResNet50" or args.model == "DenseASPP-Res101" or args.model == "DenseASPP-Res152":
    #     network = models.DenseASPP3D.build_dense_aspp(net_input, num_classes=num_classes, preset_model=args.model)
    '''
#---------------------------------------------------------------------------------------------------------------------------------------------


    if args.mode == "predict":
        for tt in range(args.startpoint, args.endpoint+1):
            print("\n***** Begin prediction *****")
            print("Dataset -->", args.pdataset)
            print("Model -->", args.model)
            print("Num Classes -->", num_classes)
            print("time point -->", tt)
            print("")

            # Create directories if needed
            addrSegRes = "/home/nirvan/Desktop/my3D_1/my3D_1/EcadMyo_08/Segmentation_Result_EcadMyo_08"
            # addrSegRes = "/home/nirvan/Desktop/my3D_1/my3D_1/EcadMyo_08/Segmentation_Result_EcadMyo_08XXXXXXXX"

            if not os.path.isdir("%s/%s/%s/%s" % (addrSegRes, args.pdataset, args.model, tt)):
                os.makedirs("%s/%s/%s/%s" % (addrSegRes, args.pdataset, args.model, tt))
            print(os.path.isdir("%s/%s/%s/%s" % (addrSegRes, args.pdataset, args.model, tt)))
            csv_path_val = '/home/nirvan/Desktop/my3D_1/my3D_1/' + args.pdataset + '/'+str(tt)+'/idx-pred.csv'
            # Path to the folder with images. Images will be read from path + path_from_csv
            path2 = csv_path_val[:csv_path_val.rfind('/')] + '/'  # + str(tt) + '/'

            df = pd.read_csv(csv_path_val)

            # Load test data
            input_image_pred, gt= loadDataGeneral(df, path2,img_size)
            print(input_image_pred[0].shape,gt[0].shape)

            # Run testing on ALL test images
            for ind in range(len(input_image_pred)):
                input_image = np.expand_dims(
                    np.float32(input_image_pred[ind]), axis=0) #/ 255.0
                print(input_image.shape)


                sys.stdout.write("\rRunning predict image %d / %d" % (ind + 1, len(input_image_pred)))
                sys.stdout.flush()

                st = time.time()
                #output_image = sess.run(network, feed_dict={net_input: input_image})
                output_image, stack = sess.run(network, feed_dict={net_input: input_image})

                #model_checkpoint_name = "checkpoints/" + args.model + "/latest_model_" + "_" + args.dataset + ".ckpt"
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
                    nib.save(new_image, "%s/%s/%s/%s/Z00%s_regression.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(input_image, "%s/%s/%s/%s/Z00%s_input.nii" % (
                    addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(final_weights_output, "%s/%s/%s/%s/Z00%s_final_weights.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                elif ind<100:
                    nib.save(new_image, "%s/%s/%s/%s/Z0%s_regression.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(input_image, "%s/%s/%s/%s/Z0%s_input.nii" % (
                    addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(final_weights_output, "%s/%s/%s/%s/Z0%s_final_weights.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                else:
                    nib.save(new_image, "%s/%s/%s/%s/Z%s_regression.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(input_image, "%s/%s/%s/%s/Z%s_input.nii" % (
                    addrSegRes, args.pdataset, args.model, tt, str(ind)))
                    nib.save(final_weights_output, "%s/%s/%s/%s/Z%s_final_weights.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))

                output_image_resize = helpers.reverse_one_hot(output_image)
                #out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
                new_image = nib.Nifti1Image(output_image_resize, affine=np.eye(4))
                if ind < 10:
                    nib.save(new_image, "%s/%s/%s/%s/Z00%s_class.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                elif ind < 100:
                    nib.save(new_image, "%s/%s/%s/%s/Z0%s_class.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))
                else:
                    nib.save(new_image, "%s/%s/%s/%s/Z%s_class.nii" % (addrSegRes, args.pdataset, args.model, tt, str(ind)))







########################################################################################################################

    elif args.mode == "test":

        print("\n***** Begin test *****")

        print("Dataset -->", args.testdataset)

        print("Model -->", args.model)

        print("Num Classes -->", num_classes)


        print("")

        # Create directories if needed

        if not os.path.isdir("%s\%s\%s" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model)):
            os.makedirs("%s\%s\%s" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model))

        # csv_path_val = 'F:/Mo/my3D_matlab/Test/idx-pred.csv'
        #
        # # Path to the folder with images. Images will be read from path + path_from_csv
        #
        # path2 = csv_path_val[:csv_path_val.rfind('/')] + '/' + str(tt) + '/'
        #
        # df = pd.read_csv(csv_path_val)
        img_size = [128, 128, 13]
        print("Loading the data from dataset: %s" % (args.testdataset))
        filename1 = glob.glob("G:/Mo/my3D_matlab/" + args.testdataset + '/*.nii')
        input_image_pred= loadDataGeneral_test_nii(filename1, img_size)
        print(input_image_pred[0].shape)


        # Run testing on ALL test images

        for ind in range(len(input_image_pred)):

            input_image = np.expand_dims(

                np.float32(input_image_pred[ind]), axis=0) #/ 255.0

            print(input_image.shape)

            sys.stdout.write("\rRunning predict image %d / %d" % (ind + 1, len(input_image_pred)))

            sys.stdout.flush()

            st = time.time()

            # output_image = sess.run(network, feed_dict={net_input: input_image})

            output_image, stack = sess.run(network, feed_dict={net_input: input_image})

            # model_checkpoint_name = "checkpoints/" + args.model + "/latest_model_" + "_" + args.dataset + ".ckpt"

            # new_saver = tf.train.import_meta_graph('/home/scw4750/Liuhongkun/tfrecord/zooscan/Alexnet/Modal/model20170226041552612/mymodel.meta')

            # run_times_list.append(time.time() - st)

            output_image = np.array(output_image[0, :, :, :])

            stack = np.array(stack[0, :, :, :, :])

            # try to accommodate the size
            w, l, h = 32, 32, 13
            # w, l, h = 32, 35, 13

            # output_image_resize = np.zeros((w, l, h))

            stack_resize = np.zeros((l, w, h, 64))
            output_resize = np.zeros((l, w, h))

            for idx in range(13):

                # img = output_image[:, :, idx, 1]

                # img_sm = cv2.resize(img, (w, l), interpolation=cv2.INTER_LINEAR)

                # output_image_resize[:, :, idx] = img_sm

                for idx2 in range(64):#stack.shape[3]
                    stk = stack[:, :, idx, idx2]

                    stk_sm = cv2.resize(stk, (w, l), interpolation=cv2.INTER_LINEAR)

                    stack_resize[:, :, idx, idx2] = stk_sm
                # output_resize[:, :, idx]=cv2.resize(output_image[:, :, idx], (w, l), interpolation=cv2.INTER_LINEAR)

            new_image = nib.Nifti1Image(output_image, affine=np.eye(4))

            final_weights_output = nib.Nifti1Image(stack_resize, affine=np.eye(4))

            if ind < 10:


                # nib.save(new_image, "%s/%s/%s/Z0000%s_regression.nii" % (
                # "F:/Mo/my3D_matlab/Test", args.testdataset, args.model, str(ind)))

                nib.save(final_weights_output, "%s/%s/%s/Z0000%s_final_weights.nii" % (
                "G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))

            elif ind < 100:

                # nib.save(new_image, "%s/%s/%s/Z000%s_regression.nii" % (
                # "F:/Mo/my3D_matlab/Test", args.testdataset, args.model, str(ind)))

                nib.save(final_weights_output, "%s/%s/%s/Z000%s_final_weights.nii" % (
                "G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))
            elif ind < 1000:

                # nib.save(new_image, "%s/%s/%s/Z00%s_regression.nii" % (
                # "F:/Mo/my3D_matlab/Test", args.testdataset, args.model, str(ind)))

                nib.save(final_weights_output, "%s/%s/%s/Z00%s_final_weights.nii" % (
                "G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))
            elif ind < 10000:

                # nib.save(new_image, "%s/%s/%s/Z0%s_regression.nii" % (
                # "F:/Mo/my3D_matlab/Test", args.testdataset, args.model, str(ind)))

                nib.save(final_weights_output, "%s/%s/%s/Z0%s_final_weights.nii" % (
                "G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))

            else:

                # nib.save(new_image,
                #          "%s/%s/%s/Z%s_regression.nii" % ("F:/Mo/my3D_matlab/Test", args.testdataset, args.model, str(ind)))

                nib.save(final_weights_output, "%s/%s/%s/Z%s_final_weights.nii" % (
                "G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))

            output_image_resize = helpers.reverse_one_hot(output_image)
            for idx in range(13):
                temp = output_image_resize[:, :, idx]
                temp=temp.astype(np.float32)
                output_resize[:, :, idx]=cv2.resize(temp, (w, l), interpolation=cv2.INTER_LINEAR)
            print(output_image_resize.shape, output_resize.shape)

            # out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            new_image = nib.Nifti1Image(output_resize, affine=np.eye(4))

            if ind < 10:

                nib.save(new_image,
                         "%s/%s/%s/Z0000%s_class.nii" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))

            elif ind < 100:

                nib.save(new_image,
                         "%s/%s/%s/Z000%s_class.nii" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))
            elif ind < 1000:

                nib.save(new_image,
                         "%s/%s/%s/Z00%s_class.nii" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))
            elif ind < 10000:

                nib.save(new_image,
                         "%s/%s/%s/Z0%s_class.nii" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))

            else:

                nib.save(new_image,
                         "%s/%s/%s/Z%s_class.nii" % ("G:/Mo/my3D_matlab/", args.testdataset, args.model, str(ind)))









