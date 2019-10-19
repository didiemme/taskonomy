from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools
import matplotlib
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import skimage.io
import transforms3d
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import utils
import models.architectures as architectures
from   data.load_ops import resize_rescale_image
from   data.load_ops import rescale_image
import utils
import lib.data.load_ops as load_ops
import task_viz

parser = utils.create_parser("Viz Single Task")
tf.logging.set_verbosity(tf.logging.ERROR)
list_of_tasks = 'autoencoder curvature denoise edge2d edge3d \
keypoint2d keypoint3d colorization jigsaw \
reshade rgb2depth rgb2mist rgb2sfnorm \
room_layout segment25d segment2d vanishing_point \
segmentsemantic class_1000 class_places inpainting_whole'
list_of_tasks = list_of_tasks.split()

def prepare_image(task, im_name, cfg):
    img = task_viz.load_raw_image_center_crop( im_name )
    img = skimage.img_as_float(img)
    scipy.misc.toimage(np.squeeze(img), cmin=0.0, cmax=1.0).save(im_name)

    if task == 'jigsaw' :
        img = cfg[ 'input_preprocessing_fn' ]( img, target=cfg['target_dict'][random.randint(0,99)], 
                                                **cfg['input_preprocessing_fn_kwargs'] )
    else:
        img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )

    img = img[np.newaxis,:]
    return img

def run_to_task():
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars

    tf.logging.set_verbosity(tf.logging.ERROR)
   
    args = parser.parse_args()

    task = args.task
    if task not in list_of_tasks:
        raise ValueError('Task not supported')

    cfg = utils.generate_cfg(task)

    # Since we observe that areas with pixel values closes to either 0 or 1 sometimes overflows, we clip pixels value
    low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
    keypoint2d keypoint3d \
    reshade rgb2depth rgb2mist rgb2sfnorm \
    segment25d segment2d room_layout'.split()
    if task in low_sat_tasks:
        cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat

    print("Doing {task}".format(task=task))
    general_utils = importlib.reload(general_utils)
    tf.reset_default_graph()
    training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

    ############## Set Up Inputs ##############
    # tf.logging.set_verbosity( tf.logging.INFO )
    setup_input_fn = utils.setup_input
    inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
    RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
    RuntimeDeterminedEnviromentVars.populate_registered_variables()
    start_time = time.time()

    ############## Set Up Model ##############
    model = utils.setup_model( inputs, cfg, is_training=False )
    m = model[ 'model' ]
    model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )

    ############## Single Image ##############

    if args.imgs_list:
        with open(args.imgs_list) as imgs_list:
            all_prediction = []
            all_representation = []

            for line in imgs_list:
                img = prepare_image(task, line.strip(), cfg) 
                predicted, representation = training_runners['sess'].run( 
                [ m.decoder_output,  m.encoder_output ], feed_dict={m.input_images: img} )

                utils.tasks(task, args, predicted, representation, img)
                all_prediction.append(np.squeeze(predicted))
                all_representation.append(np.squeeze(representation))

            if args.store_rep:
                s_name, file_extension = os.path.splitext(args.store_name)
                with open('{}.npy'.format(s_name), 'wb') as fp:
                    np.save(fp, np.array(all_representation))

            if args.store_pred:
                s_name, file_extension = os.path.splitext(args.store_name)
                with open('{}_pred.npy'.format(s_name), 'wb') as fp:
                    np.save(fp, np.array(all_prediction))
    else:
        img = prepare_image(task, args.im_name, cfg)


        predicted, representation = training_runners['sess'].run( 
                [ m.decoder_output,  m.encoder_output ], feed_dict={m.input_images: img} )

        utils.tasks(task, args, predicted, representation, img)

        if args.store_rep:
            s_name, file_extension = os.path.splitext(args.store_name)
            with open('{}.npy'.format(s_name), 'wb') as fp:
                np.save(fp, np.squeeze(representation))

        if args.store_pred:
            s_name, file_extension = os.path.splitext(args.store_name)
            with open('{}_pred.npy'.format(s_name), 'wb') as fp:
                np.save(fp, np.squeeze(predicted))
            
    ############## Clean Up ##############
    training_runners[ 'coord' ].request_stop()
    training_runners[ 'coord' ].join()
    # print("Done: {}".format(config_name))

    ############## Reset graph and paths ##############            
    tf.reset_default_graph()
    training_runners['sess'].close()
    return

if __name__ == '__main__':
    run_to_task()

