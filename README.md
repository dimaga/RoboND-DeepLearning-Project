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
`3*3*3 + 3*external_features` weights, because it consists of per-input-feature
convolution, followed by a 1x1 convolution, connecting each input feature
with each output feature of a single cell.

The middle stage consists of a number of 3x3 traditional convolutional layers
with a stride 1. This keeps the image size the same. The stage is intended to do
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
in low convergence rate and jumping around the global minimum. Too big
`batch_size` works very slow and may end up in saddle points, which are
partially compensated by Adam training algorithm.

Too low `learning_rate` may result in slow convergence rate and may require more
epochs for training. Too big `learning_rate` may jump out of the global minimum,
as well. However, actually, `learning_rate` does not play a big role in case of
Adam optimizer: usually the default value fits most of the cases.

An increase in hyper-parameter values, defining the network architecture,
increases the capacity of the network. Increased capacity may require more
training samples to avoid overfitting, more training resources and more powerful
hardware for inference.

Smaller capacity, on the other hand, may result in under-fitting, so that the
network will not generalize well both training and testing samples.

I started with the following hyper-parameter values:

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
the hero was far away from the observer.

In the simulated environment, it is expected that the network should not be very
complicated to extract pixels belonging to the hero. Basically it should only
learn how to extract dark red areas of pixels of certain shapes, which variety
is relatively small.

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

* f(x) results are noisy and not reproducible for the same x

`gp_minimize()` finds a minimum of such a function by generating some
differentiable function g(x) from a set of observations (x, f(x)), which
approximates f(x) around its minima. In my case, x is a vector of
hyper-parameters, and f(x) is implemented in  `find_train_hyperparams.py` as
`train_net()`. It returns `-final_score`, since `gp_minimize()` searches for the
minimum and I am looking for the maximum of `final_score` value.

The training procedure was run on AWS server in g2.8xlarge instance. To run
the hyper-parameter optimization procedure, I had to learn the following tricks:

1. Uninstall cpu tensor-flow in Udacity Robotics Deep Learning Laboratory, and
install tensorflow-gpu==1.2.1, which requires missing dependency on cuDNN 5.1
for CUDA 8.0. The installation of cuDNN 5.1 is explained in
https://medium.com/@ikekramer/installing-cuda-8-0-and-cudnn-5-1-on-ubuntu-16-04-6b9f284f6e77

The use of tensorflow-gpu gives 10x training performance boost as compared to
default tensorflow.

2. Extensively cache hyper-parameter optimization checkpoints to `*.pkl` files
to restart the training procedure in case the server or process crashes. 
See `CheckpointSaver` class and its use cases in `find_train_hyperparams.py`

3. Learn tmux tool that helps create a terminal session, which can be detached
from ssh-console and attached later. So that I don't keep my notebook constantly
connected to AWS console for several days of training session.

More details on tmux are available on https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session
Useful commands are:

* `tmux` - start new terminal session that can be detached from ssh
* `ctrl+b, d` - detach from the tmux terminal session
* `tmux attach` - attach to the last started tmux session
* `ctrl+b, x` - close tmux terminal session

The hyper-pararameter optimization and training took 3 days to find a solution
with final_score = 45% configuration found in 40 steps. The following values
turned out to be enough to pass the project and did not require any custom
training data:

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

Note that training results are not reproducible. From training session to
training session, the final score may be less than 40%. Therefore, I saved and
shared the results produced by the successful `find_train_hyperparams.py` run
in the shared `model_weights` file.

#### 4. The student has a clear understanding and is able to identify the use of various techniques and concepts in network layers indicated by the write-up.

Some building blocks of a neural network are shown in the picture below:

[network_parts]: ./images/network_parts.png
![alt_text][network_parts]

a) 1x1 convolution calculates an output value of each feature from different
features of a single input cell. In my case, it is used implicitly as a part of
separable convolution kernel to decrease the number of training parameters. It
could have also been used in the middle stage, instead of 3x3 convolution
layers, to further decrease the number of training parameters of the network.

Generally speaking, 1x1 convolutions shuffle information from different
features. However, they do not take into account neighbouring pixel values.
Therefore, they cannot extract information about object shape. I wouldn't use
them as the first layers of the network.

b) 3x3 convolution calculates a value of each feature of the output based on
3x3 values of all input features. 3x3 convolutions are used as a primary
building block of my network. They extract spatial information about human body
shape and surrounding context by considering colors of neighbouring pixels.

c) fully connected layer connects all the input cells to all the output cells.
The use of such a building block would significantly increase the amount of 
training parameters of my network and restrict the resolution of the input and
output image. Therefore, this building block is not used in my 
architecture. It would be a good final building block for an image
classification task, but not for a pixel-based classification.

d) bilinear upsampling increases the resolution of the image by filling up 
intermediate pixels with linearly interpolated values. The benefit of this
building block is that it does not have any training parameters. Thus, it might
work faster than deconvolution operation, which is conventionally used in this
case.

#### 5. The student has a clear understanding of image manipulation in the context of the project indicated by the write-up.

The first stage of the network pipeline is encoding. It is used to extract higher
level features of the input image, associated with color and shape.

The middle stage of the network does higher-level pixel classification by
shuffling information from neighbouring pixels and features. It is expected
to build bodies of people given body parts and surrounding information about
the context.

The last stage does decoding, increasing the resolution of the classified image
and refining its details via skip connections. Without this stage, the output
would have smaller resolution or accuracy.

#### 6. The student displays a solid understanding of the limitations to the neural network with the given data chosen for various follow-me scenarios which are conveyed in the write-up.

The simulated environment is a very easy task for classification. In fact, all
people and the main hero can be very easily distinguished by color. Therefore,
1x1 convolution layers could be enough to do the classification correctly based
on only color distribution information.

However, for real environments, the fully convolutional network would require
significantly more data and capacity. And, maybe, the use of pre-trained network
layers, such as VGG-16 or ResNet.

I see the following limitations of the fully-convolutional architecture:

* Fully convolutional networks require powerful hardware for real-time inference,
mainly due to decoding layers, which have to process a lot of data for the output

* Classes of objects similar by appearance (e.g. cats and dogs) may be
misclassified when appear in the same context (e.g. on the road) if the capacity
and amount of training data is insufficient. Of course, for other classes of
objects, we should increase the number of classes and use new data-sets where
those classes are presented and labelled.

* Per-pixel classification does not provide information about the number of
objects if their shapes intersect. This problem, called object instancing, is
feasible but requires a more sophisticated architecture

* Fully convolutional network does not utilize temporal information about the
environment when doing classification from the video. This information could be
useful to improve classification of distant objects or if the object type mainly
depends on its speed or type of motion. This is the scope of recurrent fully
convolutional network, more complex architectures.

#### 7. The model is submitted in the correct format.

`model_weights` and `config_model_weights` are available in the correct format
in `./data/weights` folder of the project. They can be loaded without errors
from `model_training.ipynb` (see also `model_training.html`) and `follower.py` with

```
python follower.py model_weights
```

command. Screenshots are available below:

[simulator_screen]: ./images/simulator_screen.png
![alt_text][simulator_screen]

[recognition]: ./images/recognition.png
![alt_text][recognition]

#### 8. The neural network must achieve a minimum level of accuracy for the network implemented.

I have achieved the final score of 0.45996904563153057, which can be seen
in the last cell of `model_training.ipynb` (see also `model_training.html`)
jupyter notebook.

#### Ideas on Futher Enhancements

* Try enhanced custom training dataset after detail  analysis of failing test
samples

* Penalize bayesian optimization algorithm for selected network architectures
with higher capacity, prefer simple architectures with minimum amount of
parameters and operations

* Learn how to cut-off hyper-parameter optimization algorithm earlier by 
analyzing the behavior of training and validation errors. Make predictions of
what the final score would be based on that informations to minimize amount of
fully training sessions

* Try deep reinforcement learning instead of bayesian optimization as more
sophisticated search for good architectures. This might require much more
hardware. Try using Google AutoML in this case (once it becomes available to
normal users).

* Add several layers of validation or testing sets to avoid overfitting caused
by hyper-parameter optimization procedure. Alternatively, increase the size of
the validation set, taking only random parts of it after each analysis.