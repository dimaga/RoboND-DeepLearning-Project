"""Searches for optimal hyper-parameter values using Bayesian Optimization
method.

The module expects the presence of skopt library, which can be installed
using the following command:

pip install scikit-optimize

See more details at https://github.com/scikit-optimize/scikit-optimize 
"""

import pickle
from skopt import gp_minimize

import os
import glob

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools
from utils import model_tools

MAX_CALLS = 100

class CheckpointSaver():
    """A wrapper of checkpoint_saver() method with iterator index to save
    different checkpoints to different files"""

    def __init__(self, iteration):
        self.__iteration = iteration


    def do(self, res):
        """Saves intermediate parameters of hyper-parameter
        optimization in case the script fails"""

        if len(res.x_iters) >= MAX_CALLS - 10:
            return

        with open("checkpoint{:08}.pkl".format(self.__iteration), "wb") as f:
            p = pickle.Pickler(f)
            p.dump(self.__iteration)
            p.dump(res.x_iters)
            p.dump(res.func_vals)

        self.__iteration += 1


def checkpoint_loader(fileName):
    """Loads hyper-parameter optimization parameters to start
    learning from previously saved checkpoint"""

    with open(fileName, "rb") as f:
        u = pickle.Unpickler(f)
        iteration = u.load()
        x_iters = u.load()
        func_vels = u.load()

    return iteration, x_iters, func_vels


def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
        activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer


def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer


def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer


def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer


def decoder_block(small_ip_layer, large_ip_layer, filters):
    upsampled = bilinear_upsample(small_ip_layer)
    concatenated = layers.concatenate([upsampled, large_ip_layer])

    output_layer1 = separable_conv2d_batchnorm(concatenated, filters)
    output_layer2 = separable_conv2d_batchnorm(output_layer1, filters)
    return output_layer2


def fcn_model(
        inputs,
        num_classes,
        layers_num,
        external_features,
        internal_features,
        conv_layers_num,
        conv_features):

    v = inputs

    layer_list = []
    for _ in range(layers_num):
        layer_list.append(v)

        if v == input:
            v = encoder_block(v, external_features, 2)
        else:
            v = encoder_block(v, internal_features, 2)

    for _ in range(conv_layers_num):
        v = conv2d_batchnorm(v, conv_features)

    for _ in range(layers_num):
        v = decoder_block(v, layer_list.pop(), internal_features)

    return layers.Conv2D(
        num_classes,
        3,
        activation='softmax',
        padding='same')(v)


steps_per_epoch = 200
validation_steps = 50
workers = 2


def train_net(x):
    learning_rate = x[0] # 0.001
    batch_size = x[1] # 32
    num_epochs = x[2] # 12
    layers_num = x[3] #2
    conv_layers_num = x[4] #1
    external_features = x[5] #128
    internal_features = x[6] #16
    conv_features = x[7] #16

    print()
    print("learning_rate", learning_rate)
    print("batch_size", batch_size)
    print("num_epochs", num_epochs)
    print("layers_num", layers_num)
    print("conv_layers_num", conv_layers_num)
    print("external_features", external_features)
    print("internal_features", internal_features)
    print("conv_features", conv_features)

    image_hw = 160
    image_shape = (image_hw, image_hw, 3)
    inputs = layers.Input(image_shape)
    num_classes = 3

    # Call fcn_model()
    output_layer = fcn_model(
        inputs,
        num_classes,
        layers_num,
        external_features,
        internal_features,
        conv_layers_num,
        conv_features)

    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy')

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(
        batch_size=batch_size,
        data_folder=os.path.join('..', 'data', 'train_combined'),
        image_shape=image_shape,
        shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(
        batch_size=batch_size,
        data_folder=os.path.join('..', 'data', 'validation'),
        image_shape=image_shape)

    model.fit_generator(
        train_iter,
        steps_per_epoch=steps_per_epoch,  # the number of batches per epoch,
        epochs=num_epochs,  # the number of epochs to train for,
        validation_data=val_iter,  # validation iterator
        validation_steps=validation_steps,  # the number of batches to validate on
        workers=workers)

    run_num = 'run_1'

    val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(
        model,
        run_num,
        'patrol_with_targ',
        'sample_evaluation_data')

    val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(
        model,
        run_num,
        'patrol_non_targ',
        'sample_evaluation_data')

    val_following, pred_following = model_tools.write_predictions_grade_set(
        model,
        run_num,
        'following_images',
        'sample_evaluation_data')

    # Scores for while the quad is following behind the target.
    true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(
        val_following,
        pred_following)

    # Scores for images while the quad is on patrol and the target is not
    # visible
    true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(
        val_no_targ,
        pred_no_targ)

    # This score measures how well the neural network can detect the target
    # from far away
    true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(
        val_with_targ,
        pred_with_targ)

    # Sum all the true positives, etc from the three datasets to get a weight
    # for the score
    true_pos = true_pos1 + true_pos2 + true_pos3
    false_pos = false_pos1 + false_pos2 + false_pos3
    false_neg = false_neg1 + false_neg2 + false_neg3

    weight = true_pos / (true_pos + false_neg + false_pos)

    # The IoU for the dataset that never includes the hero is excluded from
    # grading
    final_IoU = (iou1 + iou3) / 2

    # And the final grade score is
    final_score = final_IoU * weight

    weight_file_name = 'model_weights_' + str(final_score)
    model_tools.save_network(model, weight_file_name)
    print("Saved", weight_file_name)

    print("final_score", final_score)
    print()

    return -final_score


checkpoints = sorted(glob.glob("checkpoint*.pkl"))
if len(checkpoints) > 1:
    # The last checkpoint may be broken if the process was interrupted while
    # saving it, so load the checkpoint before the last one
    file_name = checkpoints[-2]
    start_iteration, x0, y0 = checkpoint_loader(file_name)
else:
    start_iteration = 0
    x0 = None
    y0 = None

n_calls = MAX_CALLS
if x0 is not None:
    n_calls += -len(x0)

res = gp_minimize(
    train_net,
    [(1e-7, 0.1), # learning_rate = x[0] # 0.001
     (1, 32),     # batch_size = x[1] # 32
     (5, 100),    # num_epochs = x[2] # 12
     (1, 5),      # layers_num = x[3] #2
     (1, 5),      # conv_layers_num = x[4] #1
     (4, 128),    # external_features = x[5] #128
     (4, 128),    # internal_features = x[6] #16
     (4, 128),    # conv_features = x[7] #16
    ],
    callback=[CheckpointSaver(start_iteration).do],
    x0=x0,
    y0=y0,
    n_calls=n_calls)

print(res['x'])
