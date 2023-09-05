
# CSRNet-convolution-tf
An improved implementation of CSRNet using TensorFlow, focusing on analysis of crowd patterns in densely populated images via a completely convolutional model.

## Introduction: CSRNet-convolution-tf
**CSRNet** is an innovative Deep Learning model proposed by **Yuhong Li et al.** with an objective to understand the **crowd pattern** in densely populated images. The model is a **fully convolutional** with a FrontEnd and a BackEnd.

## Implementation Aspects
In this project, we have successfully implemented all the four architectures as mentioned in the CSRNet paper. The easy-to-use **Keras API** of **TensorFlow** served as our primary tool for this implementation.

The TensorFlow's Keras API provides a multitude of high-level operations ensuring simplicity in coding without compromising the performance.
The Input pipeline employs an efficient **Data API** pipeline that parses tfRecord files.

For this, we generate a tfRecord file with all the required input data.

## About the model
The first 13 layers of the **VGGNet** pre-trained ImageNet were used as our FrontEnd Network. The BackEnd, on the other hand, is built with different combination of **Convolution** and **Dilated Convolutions** primary named as **A, B, C, and D**.

The primary Loss function for this implementation was the **simple L2 loss**, as we use smoothed label as the target.

## Training Process
We used the **Adam Optimizer** to train our model. The first 13 layers of the pre-trained VGGNet were frozen, allowing only the Backend network to be trained effectively.

Training makes use of the **ShangaiTech** Dataset and spans over 1,70,000 iterations which roughly takes around 36 hours on a system equipped with a *NVIDIA 1080Ti* graphics card and an **Intel i5 processor** paired with a 16GB RAM.

## Results
Refer to the GIFs below for a better understanding of our model's performance. The model was successful in capturing the crowd patterns in majority of the images.

**Test Images**
![](images.gif)

**Ground Truth Labels**
![](labels.gif)

**Our Predictions**
![](predictions.gif)

Continuous optimization in training can lead to better results.