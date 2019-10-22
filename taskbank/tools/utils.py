"""
    utils.py

    Contains some useful functions for creating models
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import os
import pickle
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import concurrent.futures
import argparse
import init_paths
import data.load_ops as load_ops
from   data.load_ops import create_input_placeholders_and_ops, get_filepaths_list
import general_utils
import optimizers.train_steps as train_steps
import models.architectures as architectures
from task_viz import *

def tasks(task, args, predicted, store_name, representation=None, img=None):
    if task == 'class_places' or task == 'class_1000':
        synset = get_synset(task)

    #### Multiple Imgs Tasks
    if task == 'ego_motion':
        ego_motion(predicted, store_name)
        return
    if task == 'fix_pose':
        cam_pose(predicted, store_name, is_fixated=True)
        return   
    if task == 'non_fixated_pose':
        cam_pose(predicted, store_name, is_fixated=False)
        return
    if task == 'point_match':
        prediction = np.argmax(predicted, axis=1)
        print('the prediction (1 stands for match, 0 for unmatch)is: ', prediction)
        return       
    #### Single Img Tasks
    if task == 'segment2d' or task == 'segment25d':
        segmentation_pca(predicted, store_name)
        return
    if task == 'colorization':
        single_img_colorize(predicted, img , store_name)
        return
    
    if task == 'curvature':
        curvature_single_image(predicted, store_name)
        return

    just_rescale = ['autoencoder', 'denoise', 'edge2d', 
                    'edge3d', 'keypoint2d', 'keypoint3d',
                    'reshade', 'rgb2sfnorm' ]

    if task in just_rescale:
        simple_rescale_img(predicted, store_name)
        return
    
    just_clip = ['rgb2depth', 'rgb2mist']
    if task in just_clip:
        depth_single_image(predicted, store_name)
        return
    
    if task == 'inpainting_whole':
        inpainting_bbox(predicted, store_name)
        return
        
    if task == 'segmentsemantic':
        semseg_single_image( predicted, img, store_name)
        return

    if task in ['class_1000', 'class_places']:
        classification(predicted, synset, store_name)
        return
    
    if task == 'vanishing_point':
        _ = plot_vanishing_point_smoothed(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., store_name, [])
        return
    
    if task == 'room_layout':
        mean = np.array([0.006072743318127848, 0.010272365569691076, -3.135909774145468, 
                        1.5603802322235532, 5.6228218371102496e-05, -1.5669352793761442,
                                    5.622875878174759, 4.082800262277375, 2.7713941642895956])
        std = np.array([0.8669452525283652, 0.687915294956501, 2.080513632043758, 
                        0.19627420479282623, 0.014680602791251812, 0.4183827359302299,
                                    3.991778013006544, 2.703495278378409, 1.2269185938626304])
        predicted = predicted * std + mean
        plot_room_layout(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., store_name, [], cube_only=True)
        return
    
    if task == 'jigsaw':
        predicted = np.argmax(predicted, axis=1)
        perm = cfg[ 'target_dict' ][ predicted[0] ]
        show_jigsaw((np.squeeze(img) + 1. )/2., perm, store_name)
        return

def create_parser(parser_description):
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument('--task', dest='task')

    parser.add_argument('--img', dest='im_name')

    parser.add_argument('--list', dest='imgs_list')

    parser.add_argument('--dir-name', dest='dir_name')

    parser.add_argument('--store', dest='store_name')

    parser.add_argument('--store-rep', dest='store_rep', action='store_true')
    parser.set_defaults(store_rep=False)

    parser.add_argument('--store-pred', dest='store_pred', action='store_true')
    parser.set_defaults(store_pred=False)

    parser.add_argument('--on-screen', dest='on_screen', action='store_true')
    parser.set_defaults(on_screen=False)

    return parser 

def generate_cfg(task):
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    import data.load_ops as load_ops
    from   general_utils import RuntimeDeterminedEnviromentVars
    cfg = load_config( CONFIG_DIR, nopause=True )
    RuntimeDeterminedEnviromentVars.register_dict( cfg )
    cfg['batch_size'] = 1
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join( repo_dir, 'temp', task, 'model.permanent-ckpt' )
    cfg['root_dir'] = repo_dir
    return cfg
    

def get_available_devices():
    from tensorflow.python.client import device_lib
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    return device_lib.list_local_devices()

def get_max_steps( num_samples_epoch, cfg , is_training=True):
    if cfg['num_epochs']:
        max_steps = num_samples_epoch * cfg['num_epochs'] // cfg['batch_size']
    else: 
        max_steps = None
    if not is_training:
        max_steps = num_samples_epoch // cfg['batch_size']
    print( 'number of steps per epoch:',
           num_samples_epoch // cfg['batch_size'] )
    print( 'max steps:', max_steps )
    return max_steps

def load_config( cfg_dir, nopause=False ):
    ''' 
        Raises: 
            FileNotFoundError if 'config.py' doesn't exist in cfg_dir
    '''
    if not os.path.isfile( os.path.join( cfg_dir, 'config.py' ) ):
        raise ImportError( 'config.py not found in {0}'.format( cfg_dir ) )
    import sys
    sys.path.insert( 0, cfg_dir )
    from config import get_cfg
    cfg = get_cfg( nopause )
    # cleanup
    try:
        del sys.modules[ 'config' ]
    except:
        pass
    sys.path.remove(cfg_dir)

    return cfg

def print_start_info( cfg, max_steps, is_training=False ):
    model_type = 'training' if is_training else 'testing'
    print("--------------- begin {0} ---------------".format( model_type ))
    print('number of epochs', cfg['num_epochs'])
    print('batch size', cfg['batch_size'])
    print('total number of training steps:', max_steps)


##################
# Model building
##################
def create_init_fn( cfg, model ):
    # restore model
    if cfg['model_path'] is not None:
        print('******* USING SAVED MODEL *******')
        checkpoint_path = cfg['model_path']
        model['model'].decoder
        # Create an initial assignment function.
        def InitAssignFn(sess):
            print('restoring model...')
            sess.run(init_assign_op, init_feed_dict)
            print('model restored')

        init_fn = InitAssignFn
    else:
        print('******* TRAINING FROM SCRATCH *******')
        init_fn = None
    return init_fn 

def setup_and_restore_model( sess, inputs, cfg, is_training=False ):
    model = setup_model( inputs, cfg, is_training=False )
    model[ 'saver_op' ].restore( sess, cfg[ 'model_path' ] )
    return model

def setup_input( cfg, is_training=False, use_filename_queue=False ):
    '''
        Builds input tensors from the config.
    '''
    inputs = {}
    # Generate placeholder input tensors
    placeholders, batches, load_and_enqueue, enqueue_op = create_input_placeholders_and_ops( cfg )
    
    input_batches = list( batches ) # [ inputs, targets, mask, data_idx ]
    
    inputs[ 'enqueue_op' ] = enqueue_op
    inputs[ 'load_and_enqueue' ] = load_and_enqueue
    inputs[ 'max_steps' ] = 6666 
    inputs[ 'num_samples_epoch' ] = 6666 

    inputs[ 'input_batches' ] = input_batches
    inputs[ 'input_batch' ] = input_batches[0]
    inputs[ 'target_batch' ] = input_batches[1]
    inputs[ 'mask_batch' ] = input_batches[2]
    inputs[ 'data_idxs' ] = input_batches[3]
    inputs[ 'placeholders' ] = placeholders
    inputs[ 'input_placeholder' ] = placeholders[0]
    inputs[ 'target_placeholder' ] = placeholders[1]
    inputs[ 'mask_placeholder' ] = placeholders[2]
    inputs[ 'data_idx_placeholder' ] = placeholders[3]
    return inputs


def setup_model( inputs, cfg, is_training=False ):
    '''
        Sets up the `model` dict, and instantiates a model in 'model',
        and then calls model['model'].build

        Args:
            inputs: A dict, the result of setup_inputs
            cfg: A dict from config.py
            is_training: Bool, used for batch norm and the like

        Returns:
            model: A dict with 'model': cfg['model_type']( cfg ), and other
                useful attributes like 'global_step'
    '''
    validate_model( inputs, cfg )
    model = {}
    model[ 'global_step' ] = slim.get_or_create_global_step()

    model[ 'input_batch' ] = tf.identity( inputs[ 'input_batch' ] )
    if 'representation_batch' in inputs:
        model[ 'representation_batch' ] = tf.identity( inputs[ 'representation_batch' ] )
    model[ 'target_batch' ] = tf.identity( inputs[ 'target_batch' ] )
    model[ 'mask_batch' ] = tf.identity( inputs[ 'mask_batch' ] )
    model[ 'data_idxs' ] = tf.identity( inputs[ 'data_idxs' ] )

    # instantiate the model
    if cfg[ 'model_type' ] == 'empty':
        return model
    else:
        model[ 'model' ] = cfg[ 'model_type' ]( global_step=model[ 'global_step' ], cfg=cfg )

    # build the model
    if 'representation_batch' in inputs:
        input_imgs = (inputs[ 'input_batch' ], inputs[ 'representation_batch' ])
    else:
        input_imgs = inputs[ 'input_batch' ]
    model[ 'model' ].build_model( 
            input_imgs=input_imgs,
            targets=inputs[ 'target_batch' ],
            masks=inputs[ 'mask_batch' ],
            is_training=is_training )
    
    if is_training:
        model[ 'model' ].build_train_op( global_step=model[ 'global_step' ] )
        model[ 'train_op' ] = model[ 'model' ].train_op
        model[ 'train_step_fn' ] = model[ 'model' ].get_train_step_fn()
        model[ 'train_step_kwargs' ] = train_steps.get_default_train_step_kwargs( 
            global_step=model[ 'global_step' ],
            max_steps=inputs[ 'max_steps' ],
            log_every_n_steps=10 )

    #model[ 'init_op' ] = model[ 'model' ].init_op
    if hasattr( model['model'], 'init_fn' ):
        model[ 'init_fn' ] = model['model'].init_fn
    else:
        model[ 'init_fn' ] = None

    max_to_keep = cfg['num_epochs'] * 2
    if 'max_ckpts_to_keep' in cfg:
        max_to_keep = cfg['max_ckpts_to_keep']
    model[ 'saver_op' ] = tf.train.Saver(max_to_keep=max_to_keep)
    return model

def validate_model( inputs, cfg ):
    general_utils.validate_config( cfg )



