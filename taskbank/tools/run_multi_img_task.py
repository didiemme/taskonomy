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


parser = utils.create_parser("Viz Multiple Task")
tf.logging.set_verbosity(tf.logging.ERROR)
list_of_tasks = 'ego_motion \
fix_pose \
non_fixated_pose \
point_match'
list_of_tasks = list_of_tasks.split()


def run_to_task():
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars

    tf.logging.set_verbosity(tf.logging.ERROR)
   
    args = parser.parse_args()

    imgs = args.im_name.split(',')

    if args.task == 'ego_motion' and len(imgs) != 3:
        raise ValueError('Wrong number of images, expecting 3 but got {}'.format(len(imgs)))
    if args.task != 'ego_motion' and len(imgs) != 2:
        raise ValueError('Wrong number of images, expecting 2 but got {}'.format(len(imgs)))

    
    task = args.task
    if task not in list_of_tasks:
        raise ValueError('Task not supported')

    cfg = utils.generate_cfg(task)

    input_img = np.empty((len(imgs),256,256,3), dtype=np.float32)
    for i,imname in enumerate(imgs):
        img = load_raw_image_center_crop( imname )
        img = skimage.img_as_float(img)
        scipy.misc.toimage(np.squeeze(img), cmin=0.0, cmax=1.0).save(imname)
        img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )
        input_img[i,:,:,:] = img
    input_img = input_img[np.newaxis, :]
    

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

    predicted, representation = training_runners['sess'].run( 
            [ m.decoder_output,  m.encoder_output ], feed_dict={m.input_images: input_img} )

    utils.tasks(task, args, predicted, args.store_name)

    
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

