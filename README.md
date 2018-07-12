## Deep Learning Project ##

My solution of RoboND-DeepLearning-Project assignment from Udacity Robotics Nanodegree
course, Term 1. See project assignment starter code in
https://github.com/udacity/RoboND-DeepLearning-Project

---


## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

#### 1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner.

You're reading it!

#### 2. The write-up conveys the an understanding of the network architecture.

`fcn_model()` from `find_train_hyperparams.py` and `model_training.ipynb`
creates a fully convolutional network to classify pixels belonging to people
and to the hero in the input image.

The network architecture is shown in the picture below.

[network_architecture]: ./images/network_architecture.png
![alt_text][network_architecture]

The first stage consists of a number of separable 3x3 convolution layers with
a stride 2. Each new layer is intended to extract higher level features from
the image,  decreasing the image resolution (because of the stride > 1). Custom
implementation of separable convolution from `separable_conv2d.py` helps to
keep the number of training parameters smaller. For example, traditional
convolution would require `3*3*3*external_features` weights for the
first layer of the network. However, separable convolution requires only
`3*3*3 + 3*external_features` weights, because it consists of per-input
feature convolution, followed by 1x1 convolution, connecting each input feature
with each output feature of a single cell.

The middle stage consists of a number of 3x3 traditional convolutional layers
with stride 1. This keeps the image size the same. The stage is intended to do
higher level pixel classification analysis.

The last stage restores the resolution of the output by interleaving biinear
up-scaling operation and separable 3x3 convolution with stride 1. To add details
to the up-scaled image, skip-connections from the first stage layers are
concatenated to the output of each up-scaling operation.
 
The fully convolutional network has the following benefits over a fully connected
network:

* It is capable of pixel-level classification

* It does not depend on the resolution of the input and the output

* All its parameters are shared across different parts of the image. This
decreases the amount of parameters and lets it learn faster, generalizing
information more efficiently and being agnostic of translational 2D
transformations

The number of layers in each stage, as well as the number of features are
defined by the following hyper-parameters:

1. `layers_num` defines the number of layers in the first and the last stage

2. `conv_layers_num` defines the number of layers in the middle stage,
consisting of traditional convolutional layers

3. `external_features` defines the number of features in the first layer output

4. `internal_features` defines the number of features in the first and the last
stage, except for the first layer output

5. `conv_features` defines the number of features in the middle stage of the
network; the output of 3x3 traditional convolutional layers

I combine these hyper-parameters with those that define the training procedure.


#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

In addition to hyper-parameters, defining the network architecture, the
following hyper-parameters, defining the training procedure were tuned:

* `num_epochs` defines how many times the whole training set is scanned

* `batch_size` defines how many elements from the training set are taken to
calculate partial derivatives of stochastic gradient descent when training

* `learning_rate` defines the step size of the gradient descent. Adam optimizer,
which is used in the project, does not directly use this hyper-parameter when
adjusting training parameters. Instead, it adjusts the actual step size
dynamically based on the history of convergence rate. The given `learning_rate`
is used inside Adam optimizer to update its own parameters.

Because the search of optimal training parameters is non-linear, the process
requires several `num_epochs`. Too many `num_epochs` results in overfitting
when neural network shows low error in training data set, but big error in
validation or test sets.

Too small `batch_size` works faster and helps avoid local minima, but may result
in low convergence rate and jumping around global minima. Too big `batch_size`
works very slow and may end up in saddle points, which are partially compensated
by Adam training algorithm.

Too low `learning_rate` may result in slow convergence rate and may require more
epochs for training. Too big `learning_rate` may jump out of global minima, as
well.

Increase in hyper-parameter values, defining the network architecture, increases
the capacity of the network. Increased capacity may require more training
samples to avoid overfitting, more training resources and more powerful hardware
for inference.

Smaller capacity, on the other hand, may result in under-fitting, so that the
network will not generalize well both traning and testing samples.

I started with the following parameters first:

```
learning_rate = 0.001
batch_size = 32
num_epochs = 12
layers_num = 2
conv_layers_num = 1
external_features = 128
internal_features = 16
conv_features = 16
```

This produced 25% final score. Most of the mistakes came from the cases where
the hero was far from the observer.

In the simulated environment, it is expected that the network should not be very
complicated to extract pixels belonging to the hero. Basically it should only
learn how to extract dark red areas of pixels of a certain shape, which variety
is not too big.

However, having previous experience with manual adjustments of hyper-parameters
in https://github.com/dimaga/CarND-Semantic-Segmentation project, I decided to
refrain from this tedious process this time, and try something new.

I ended up with the Bayesian Optimization procedure, which has become more
mature for the past few years. Sk-opt library from
https://scikit-optimize.github.io suggests a number of algorithms to
automatically search the minimum of some function f(x), which has the following
properties:

* Calculation of f(x) is very slow

* There is no derivative of f(x) or it is very difficult to calculate

* f(x) result is noisy and not reproducible

`gp_minimize()` allows to find a minimum of this function by generating some
differentiable function g(x) from a set of observations (x, f(x)). In my case,
x is a vector of hyper-parameters, and f(x) is implemented in 
`find_train_hyperparams.py` as `train_net()`. It returns `-final_score`, since
`gp_minimize()` searches for the minimum and I am looking for the maximum of
`final_score` value.

The training procedure was run on AWS server in g2.8xlarge instance. To run
the hyper-parameter optimization procedure, I had to make the following steps:

1. Uninstall cpu tensor-flow in Udacity Robotics Deep Learning Laboratory, and
install tensorflow-gpu==1.2.1, which requires missing dependency on cuDNN 5.1
for CUDA 8.0. The installation of cuDNN 5.1 is explained in
https://medium.com/@ikekramer/installing-cuda-8-0-and-cudnn-5-1-on-ubuntu-16-04-6b9f284f6e77

The use of tensorflow-gpu gives 10x training performance boost

2. Extensively cache hyper-parameter optimization checkpoints to `*.pkl` files
to restart the training procedure in case it fails. See `CheckpointSaver` class
and its use cases in `find_train_hyperparams.py`

3. Learn tmux tool that helps create a terminal session, which can be detached
from ssh-console and attach later. So that I don't keep my notebook constantly
connected to AWS console for a several days of training session.

The hyper-pararameter optimizer took 3 days to find a solution with
final_score = 45% in 40 steps. The following values turned out to be optimal and
did not require any additional training data:

```
learning_rate = 0.01
batch_size = 23
num_epochs = 75
layers_num = 1
conv_layers_num = 3
external_features = 64
internal_features = 32
conv_features = 23
```

Note that results are not reproducible. From training session to training
session, the final score may be less than 40%. Therefore, I saved and shared
the results produced by the sucessful `find_train_hyperparams.py` run in the
given `model_weights`.

#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

[network_parts]: ./images/network_parts.png
![alt_text][network_parts]

The student demonstrates a clear understanding of 1 by 1 convolutions and where/when/how it should be used.

The student demonstrates a clear understanding of a fully connected layer and where/when/how it should be used.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The student is able to identify the use of various reasons for encoding / decoding images, when it should be used, why it is useful, and any problems that may arise.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

#### 7. The student is able to clearly articulate whether this model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

The file is in the correct format (.h5) and runs without errors.

[simulator_screen]: ./images/simulator_screen.png
![alt_text][simulator_screen]

[recognition]: ./images/recognition.png
![alt_text][recognition]

#### 8. The neural network must achieve a minimum level of accuracy for the network implemented.

The neural network should obtain an accuracy greater than or equal to 40% (0.40) using the Intersection over Union (IoU) metric.

#### Ideas on Futher Enhancements