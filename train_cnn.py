# USAGE
# python run_mnist_training.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# import the necessary packages
from models import GenericNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow import device
import tensorflow as tf

import utils
import matplotlib.pyplot as plt
import numpy as np
import pickle
import gc

from pathlib import Path
from datetime import datetime
import shutil

from config import shapes_config as config


def parse_command_line_args(args):
    if args['smoothing'] is None:
        # default smoothing parameters
        sf_list = [0.5, 0.6, 0.7]
    else:
        assert 1 > args['smoothing'] >= 0, 'smoothing parameter must be in [0,1)'
        sf_list = [args['smoothing']]

    if args['trait'] is None:
        # default to exhaustive list of trait combinations
        experiment_traits = ['scale', 'color', 'shape', 'all', 'color-shape', 'color-size', 'shape-size']
    else:
        possible_traits = ['scale', 'color', 'shape', 'all', 'color-shape', 'color-size', 'shape-size']
        assert args['trait'] in possible_traits, 'trait not recognized.'
        experiment_traits = [args['trait']]

    # defaults to [0.5, 0.5] for equal trait weighting
    assert 1 > args['dualweight'] > 0, 'dualweight parameter must be in (0,1)'
    trait_weights = [args['dualweight'], 1-args['dualweight']]

    return sf_list, trait_weights, experiment_traits

# to fix cudnn handle error
utils.train.configure_gpu_options()
args = utils.train.get_command_line_args(run_type='train')

# this forces the same weights for all dual trait combinations with the trait_weights variable. 
# For more flexibility you can redefine it as a custom dictionary with the respective trait 
# combinations as keys, i.e. 'color-shape', 'color-size', and 'shape-size', and the values each 
# being a 2d list of values in (0,1) that sum to 1 e.g. - 
# trait_weights = {'color-shape': [0.2, 0.8], 'color-size': [0.4,0.6], 'shape-size': [0.5,0.5]}
# the variable will be overriden 
sf_list, trait_weights, experiment_traits = parse_command_line_args(args)

input_shape = config.DATASET_PARAMS['input_shape']
num_classes = config.DATASET_PARAMS['num_classes']
epochs, init_lr, batch_size, verbose = config.get_default_training_config()

# any of the number of epochs, learning rate, or batch_size for training could be
# changed by hardcoding the above variables here to a different value
epochs = 50

# load model parameters
if args['params'] is None:
    model_params = utils.train.load_default_model_params()
else:
    with open(args['params'], 'rb') as f:
        model_params = pickle.load(f)

# should be either "linear" for smoothed labels that reflect a potential continuum of perceptual
# similarity or "coarse" to enforce binary relations across perceptual categories
label_type = 'linear'

# replace with correct path for the tensorflow shapes3d dataset
# for download instructions visit https://github.com/deepmind/3d-shapes
# can also pass the datapath via command line with the -d option
datapath = args['datapath']

print("[INFO] loading data...")
train_data, validation_data, target_names, full_labels = utils.train.load_data(input_shape,
                                                                               balance_traits=True,
                                                                               label_type=label_type,
                                                                               return_full_labels=True,
                                                                               datapath=datapath)
num_classes = 64
smooth_func = utils.train.sum_label_components

date = datetime.today()
date_str = date.strftime('%y%m%d')

# currently calculates labels for all settings, which is not necessary if you
# only want a subset of the trait combinations. If you want to customize what types
# of relationships you are enforcing in the vision module, this is the function to 
# take a look at. Essentially any vectors with the right dimensionality that sum to one
# will work for training, but how you structure those vectors will determine what class 
# relationships are being enforced
_, relational_labels, _, trait_weights = utils.train.get_shape_color_labels(full_labels,
                                                                            balance_traits=True,
                                                                            balance_type=2,
                                                                            label_type=label_type,
                                                                            trait_weights=trait_weights)

for trait_to_enforce in experiment_traits:
    print('enforcing trait: {}'.format(trait_to_enforce))

    if trait_to_enforce in ['color-shape', 'color-size', 'shape-size']:
        tw = trait_weights[trait_to_enforce]
    else:
        tw = None

    for FACTOR in sf_list:
        with device('/gpu:' + str(args["gpu"])):
            opt = SGD(learning_rate=init_lr, momentum=0.9)

            # the line below could be exchanged for any keras model with appropriate input/output
            # characteristics for the shapes3d dataset. The GenericNet class is also relatively
            # convenient for building vanilla CNNs, if you take a few minutes to understand how to
            # set the model_params dictionary
            model = GenericNet.build(*input_shape, num_classes, **model_params)
            model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

            # sets checkpoint paths designed for multiple experimental runs by creating a folder based
            # on the date of the experiment
            checkpoints_path = os.path.join(args['checkpoints'],
                                            date_str, label_type, trait_to_enforce,
                                            'tw-', str(tw)[-2:] + '_sf-' + str(FACTOR).replace('.', '-'))
            if not os.path.isdir(checkpoints_path):
                Path(checkpoints_path).mkdir(parents=True)

            if args['params'] is not None:
                shutil.copyfile(args['params'], os.path.join(checkpoints_path, 'model_params.pkl'))

            fname = os.path.sep.join([checkpoints_path,
                                      "weights-sfactor-{:.2f}".format(FACTOR) + "-{epoch:03d}-{val_loss:.4f}.hdf5"])

            callbacks = [ModelCheckpoint(fname,  monitor='val_loss', save_freq='epoch')]

            # ModelTrainer class expects training arguments as a dictionary
            training_args = dict()
            param_kws = ('validation_data', 'batch_size', 'epochs', 'callbacks', 'verbose')
            for kw in param_kws:
                training_args[kw] = locals()[kw]
            sf_args = {'factor': FACTOR,
                       'verbose' : True,
                       'rel_comps' : relational_labels[trait_to_enforce]}
            mt = utils.ModelTrainer(model, train_data,
                                    train_args=training_args,
                                    eval_args={'batch_size':batch_size},
                                    sfunc=smooth_func,
                                    verbose=True,
                                    func_args=sf_args)

            save_vars = (mt.H.history, mt.report, mt.c_matrix, relational_labels, tw, trait_to_enforce)
            fn_out = os.path.sep.join(
                [checkpoints_path, 'exp_data_sf-{:.2f}-trait-{}.pkl'.format(FACTOR, trait_to_enforce)])

            # this code saves some variables that are useful for analysis and then also saves the final
            # weights files in a separate directory for ease of access. Could theoretically be deleted
            # and just use the checkpoints files being saved by the training callback
            with open(fn_out, 'wb') as f:
                pickle.dump(save_vars, f)
            finalEpoch_dir = os.path.join(args['checkpoints'], date_str, 'final_epochs')
            if trait_to_enforce in ['color-shape', 'color-size', 'shape-size']:
                finalEpoch_fn = 'finalweights_lt-' + label_type + '_trait-' + trait_to_enforce + \
                        '_tw-' + '{:.02f}'.format(tw[0])[-2:] + '_sf-' + str(FACTOR) + '.hdf5'
                dataFN_out = os.path.join(
                    finalEpoch_dir, 'expdata_tw-{:.2f}_sf-{:.2f}_trait-{}.pkl'.format(tw[0], FACTOR,trait_to_enforce))
            else:
                finalEpoch_fn = 'finalweights_lt-' + label_type + '_trait-' + trait_to_enforce + \
                        '_sf-' + str(FACTOR) + '.hdf5'
                dataFN_out = os.path.join(
                    finalEpoch_dir, 'expdata_trait-{}_sf-{:.2f}.pkl'.format(trait_to_enforce, FACTOR))
            if not os.path.isdir(finalEpoch_dir):
                Path(finalEpoch_dir).mkdir(parents=True)
            model.save(os.path.join(finalEpoch_dir, finalEpoch_fn))
            with open(dataFN_out, 'wb') as f:
                pickle.dump(save_vars, f)

            # if you don't do this you may run into memory issues when training multiple models in a
            # single run
            tf.keras.backend.clear_session()
            gc.collect()

            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, epochs), mt.H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, epochs), mt.H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, epochs), mt.H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, epochs), mt.H.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(os.path.join(checkpoints_path, 'training_plot.png'))
            plt.close()
