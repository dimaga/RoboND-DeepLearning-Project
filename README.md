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

The network architecture is shown in the picture below. It consists of a
number of separable 3x3 convolution layers with stride 2 that decreases the size
of the input image, increasing the number of features. The next stage is a
number of 3x3 convolution layers with stride 1 that keep the image size the same
but may change the number of features. The last stage interleaves bilinear
up-scaling operation with concatenation and separable 3x3 convolution with
stride 1. This increases the size of the image up to the original, adding
details from skip-connected layers.

[network_architecture]: ./images/network_architecture.png
![alt_text][network_architecture]

Such an architecture, which does not include fully connected layers, is called
a fully convolutional network. Its benefits are:

* It is capable of pixel-level classification

* It does not depend on the resolution of the input and the output

* All its parameters are shared across different parts of the image. This
decreases the amount of parameters and lets it learn faster, generalizing
information more efficiently

The number of layers in each stage, as well as the number of features are
defined by additional hyper-parameters:

1. layers_num
2. conv_layers_numb
3. external_features
4. internal_features
5. conv_features

I combine these hyper-parameters with those that define the training pace.
Instead of manual adjustments, optimal values of hyper-parameters are
automatically searched with bayesian optimization procedure, implemented in
skopt library.

#### 3. The write-up conveys the student's understanding of the parameters chosen for the the neural network.

The student explains their neural network parameters including the values selected and how these values were obtained (i.e. how was hyper tuning performed? Brute force, etc.) Hyper parameters include, but are not limited to:

Epoch
Learning Rate
Batch Size
Etc.
All configurable parameters should be explicitly stated and justified.

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