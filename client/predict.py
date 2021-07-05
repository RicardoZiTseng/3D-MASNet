import argparse
import os
import time

import keras.backend as K
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from models.dunet import DUNetMixACB
from util.metrics import dice
from util.predict_funcs import predict_segmentation
from util.prep import image_norm, maybe_mkdir_p, vote
from util.timer import Clock

def save(t1_path, save_path, segmentation):
    t1_affine = nib.load(t1_path).get_affine()
    segmentation = np.array(segmentation, dtype=np.uint8)
    seg = nib.AnalyzeImage(segmentation, t1_affine)
    nib.save(seg, save_path)

def analyze_score(dsc_score):
    csf = []
    gm = []
    wm = []
    for i in range(len(dsc_score)):
        csf.append(dsc_score[i][0])
        gm.append(dsc_score[i][1])
        wm.append(dsc_score[i][2])
    print('%s   | %2.2f | %2.2f | %2.2f | %2.2f |' % ('Avg Dice', np.mean(csf), np.mean(
        gm), np.mean(wm), np.mean([np.mean(csf), np.mean(gm), np.mean(wm)])))


def cal_acc(label, pred):
    dsc = []
    print('------------------------------------------')
    for i in range(1, 4):
        dsc_i = dice(pred, label, i)
        dsc_i = round(dsc_i*100, 2)
        dsc.append(dsc_i)
    print('Data     | CSF   | GM    | WM    | Avg.  |')
    print('%s   | %2.2f | %2.2f | %2.2f | %2.2f |' %
          ('Dice', dsc[0], dsc[1], dsc[2], np.mean(dsc)))
    return dsc

def predict(path_dict, model, cube_size, strides):
    affine = nib.load(path_dict['t1w']).get_affine()
    t1_data = nib.load(path_dict['t1w']).get_data()
    t2_data = nib.load(path_dict['t2w']).get_data()
    vol_size = t1_data.shape[0:3]
    mask = (t1_data > 0)

    t1_data_norm = image_norm(t1_data)
    t2_data_norm = image_norm(t2_data)

    subject_data = {'t1w': t1_data_norm, 't2w': t2_data_norm}

    segmentation = predict_segmentation(
        subject_data, 30, cube_size, strides, model, 1)
    
    segmentation = np.expand_dims(np.argmax(segmentation, axis=-1), axis=-1)

    if 'label' in path_dict.keys():
        label_data = nib.load(path_dict['label']).get_data()
        cal_acc(label_data, segmentation)

    return segmentation

def main(params):
    gpu_id = params.gpu_id
    save_folder = params.save_folder
    data_path = params.data_path
    cube_size = [32] * 3
    strides = [8] * 3
    clock = Clock()

    maybe_mkdir_p(os.path.abspath(save_folder))

    # close useless warning information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    with tf.device(gpu):
        normal_model = DUNetMixACB(deploy=params.deploy).build_model()

        dice_score = []

        for i in params.subjects:
            
            print("preprocess subject: {}".format(i))
            t1_path = os.path.join(data_path, 'subject-' + str(i) + '-T1.img')
            t2_path = os.path.join(data_path, 'subject-' + str(i) + '-T2.img')
            save_seg_path = os.path.join(
                save_folder, 'subject-' + str(i) + '-label.img')
            input_dict = {'t1w': t1_path, 't2w': t2_path}
            segmentation_sets = []
            if params.predict_mode == 'evaluation':
                input_dict['label'] = os.path.join(
                    data_path, 'subject-' + str(i) + '-label.img')
            for j in range(len(params.model_files)):
                print("subj {}, j {}".format(i, j))
                normal_model.load_weights(params.model_files[j])
                clock.tic()
                seg = predict(input_dict, normal_model, cube_size, strides)
                segmentation_sets.append(seg)
            final_seg = vote(segmentation_sets)
            save(t1_path, save_seg_path, final_seg)

            clock.toc()
            if params.predict_mode == 'evaluation':
                print("The ensemble result is :")
                label_data = nib.load(input_dict['label']).get_data()
                dsc = cal_acc(label_data, final_seg)
                dice_score.append(dsc)
        print("The average time is {}.".format(clock.average_time/60))
        
        if params.predict_mode == 'evaluation':
            analyze_score(dice_score)
