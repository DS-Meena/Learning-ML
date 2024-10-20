# Image Convolution

In this project, we will create two small programs to compare different Image filters and Pooling methods.

# Convolution Layers

### Convolution operation

Suppose we have N x N matrices A and B. We define $A \circ B$ as follows, where $\circ$ is the convolutional operator. 

$(A \circ B)_{i,j} = A_{i,j}B_{i,j}$

The activation of a filter on an image of the same size is found by applying the convolutional operator to the filter and the image, then taking the sum of the entries in the resulting matrix, adding the filter's bias, and applying an activation function to it. 

### Striding (to walk)

If we have a filter, we can apply it to a larger matrix (image) at a bunch of possible positions.

If we have an $N × N$  filter and an $M × M$ image, in how many positions can we apply the filter? $(M - N + 1)^2$

there will be (M-N+1) positions along both axes.

We can also skip spaces when striding by specifying the number of squares to move when shifting our filter.

Then number of positions would be = $(\frac {W - N}{S} + 1) * (\frac {H - N}{S} + 1)$

**Padding**

We pad the image on sides so that we can catch partial objects.

![Fig: applied p=1 on image](https://prod-files-secure.s3.us-west-2.amazonaws.com/5989232b-4798-4c63-ac9f-04cb2f5fb1a8/13d81d66-83ce-43aa-8a5d-83af193b8667/Untitled.png)

Fig: applied p=1 on image

What will be the dimensions of the feature map for a filter with size $N × N$ applied to an image with size $W × H$ with padding P and stride length S?

Dimensions of feature map or number of positions, where we apply our filter: 

$(\frac {W - N + 2P}{S} + 1) * (\frac {H - N + 2P}{S} + 1)$

**Using multiple convolutional step**

![Fig: Feature matrix at first step](https://prod-files-secure.s3.us-west-2.amazonaws.com/5989232b-4798-4c63-ac9f-04cb2f5fb1a8/baddf158-0733-46c9-858b-fcdf0f98e6d3/Untitled.png)

Fig: Feature matrix at first step

When actually creating a CNN, we will want to have multiple filters to recognize different features at each convolutional step.

This means that our output matrix has a **volume**: if we have K $N × N$ filters and apply them to a $W × H$ image with P padding and S stride length, our output matrix will have dimensions $(\frac{W−N+2P}{S}+1)×(\frac{H−N+2P}{S}+1)×K$.

So, if we want to use another convolution on the matrix with depth K above, we will need to use filters that also have depth K.

Example → 

First step: we have applied three $5 × 5$ filters to a $99 × 99$ image with a stride of 1 and padding of 3, and gotten a matrix M1 (size = 101 x 101 x 3). Using above formula

Suppose that we want to apply another convolution to M1 with a stride of 2 and padding of 5, and we want the dimensions of this output matrix M2 to be $50 × 50 × 6$.

**Solution:** 

old depth = 3, old size = 101 x 101

Since, new depth = 6 hence filters = 6

using $(\frac{101−x+2⋅5}{2}+1)=50$ ⇒ $x = 13$

(size = 13 x 13 x 3)

Image convolution is a image filtering technique. 

Neurons in first convolutional layer are not connected to every single pixel in the input image, but only to pixels in their receptive fields. This architecture allows the network to concentrate on small low-level features in the first hidden layer, then assemble them into larger higher-level features in the next hidden layer, and so on

![Fig: Connection between layers and zero padding](https://prod-files-secure.s3.us-west-2.amazonaws.com/5989232b-4798-4c63-ac9f-04cb2f5fb1a8/b3f2c3a2-e090-4a8f-bba9-69dfccbd785f/Untitled.png)

Fig: Connection between layers and zero padding

(i, j) neuron of a given layer is connected to $(i, i+f_h-1) * (j, j+f_w-1)$ neurons of previous layer.

It is also possible to connect a large input layer to a much smaller layer by spacing out the receptive fields. The shift from one receptive field to the next is called the stride.

(i, j) neuron is connected to $(i*s_h, i*s_h + f_h - 1) * (j*s_w, j*s_w + f_w - 1)$

$s_h, s_w$ are vertical and horizontal strides.

![Fig: Reducing dimensionality using a stride of 2](https://prod-files-secure.s3.us-west-2.amazonaws.com/5989232b-4798-4c63-ac9f-04cb2f5fb1a8/1fae4261-4400-441b-8886-45e8c6ea4f4b/Untitled.png)

Fig: Reducing dimensionality using a stride of 2

Stacked convolutional layers

$z_{i,j,k} = b_k + \sum_{u=0}^{f_h-1}\sum_{v=0}^{f_w - 1}\sum_{k'=0}^{f_{n'}-1}x_{i',j',k'}.w_{u,v,k',k}$

$i’ = i*s_h + u$

$j’=j*s_w + v$

You can also define filters as trainable variables:

```python
conv = keras.layers.Conv2D(filter=32, kernel_size=3, strides=1, 
													padding="same", activation="relu")
```

A CNN layer contains many kernels.

## Troubleshooting

```
2024-10-20 12:35:17.323136: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-10-20 12:35:17.354318: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-20 12:35:17.591468: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2024-10-20 12:35:17.888093: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-10-20 12:35:18.125883: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-10-20 12:35:18.190703: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-10-20 12:35:18.570909: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-10-20 12:35:21.157449: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
```

Disable the warnings using following commands:
```
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

To again enable the warnings use following commands:
```
unset TF_ENABLE_ONEDNN_OPTS
export TF_CPP_MIN_LOG_LEVEL=0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
```