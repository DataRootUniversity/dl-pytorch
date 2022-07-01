# Deep Learning Project &mdash; Part 2: New Models
*Hello and welcome back to the last frontier of your course!*

*During this project section, you will learn and define new models such as Fast RCNN and Mask R-CNN and integrate them to the pipeline you've done before.*

## Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Fast SCNN](#fast-scnn)
    -  [Defining Fast SCNN model](#defining-fast-scnn-model)
    -  [Defining Pytorch Lightning Fast SCNN trainer](#defining-pytorch-lightning-fast-scnn-trainer)
    -  [Defining Fast SCNN Inference](#defining-fast-scnn-inference)
-  [Mask R-CNN](#mask-r-cnn)
    -  [Defining Mask R-CNN model](#defining-mask-r-cnn-model)
    -  [Defining Pytorch Lightning Mask R-CNN trainer](#defining-pytorch-lightning-mask-r-cnn-trainer)
    -  [Defining Mask R-CNN Inference](#defining-mask-r-cnn-inference)
-  [Models Training and Inference](#models-training-and-inference)
    -   [Models Training](#models-training)
    -   [Models Inference](#models-inference)
-  [Conclusion](#conclusion)

## Intro

As it was said above, you gonna implement Fast SCNN and Mask R-CNN models:
- The Fast SCNN (Fast Segmentation Convolutional Neural Network) is a model which solves a semantic image segmentation task. The model containst only 1.1M parameters, thus, the training and inference are done pretty fast and computationally cheap.
- The Mask R-CNN is a Convolutional Neural Network with 44M parameters which detects objects in an image and generates a segmentation mask for each instance. In other words, it can separate different objects in an image.

In general, segmentation task is devided into semantic and instance segmentation tasks. Semantic segmentation associates every pixel of an image with a class label such as a person, dog, car, etc. It treats multiple objects of the same class as a single entity, so it's just like UNet and Fast SCNN do. In contrast, the Mask R-CNN, as instance segmentation model, treats multiple objects of the same class as distinct individual instances.

Let’s consider an image below with apples and orange:

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/segm_masks.png">
</div>


Semantic Segmentation can mark out the apples’ and orange pixels. However, there is no indication of how many different objects are in the image. With Instance Segmentation, we can find the bounding boxes of each instance as well as the segmentation maps for each object. Also, we can know the number of instances in the image. 

Notice that the dataset for this project contains only the pictures **depicting a single car**. In other words, every pixel of every image is binary labelled. Thus, semantic and instance segmentation tasks are similar in this case. However, the Mask R-CNN model may recognize and bound multiple cars on a picture as separate instances **even being trained on single-car images**. 

## Project Structure

Will slightly expand the previous project structure adding a new models in the `scr/models` folder:
```
DLProject
        ├── data                                       - contains full data with images and their masks.       
        │    ├── images                                - contains all images.       
        │    │     ├── 0cdf5b5d0ce1_01.jpg
        │    │     ├── 0cdf5b5d0ce1_02.jpg
        │    │     ├── ...
        │    │
        │    └── masks                                 - contains all ground truth masks.
        │          ├── 0cdf5b5d0ce1_01.png
        │          ├── 0cdf5b5d0ce1_02.png  
        │          ├── ...
        │
        ├── experiments                                - contains trained models, checkpoints, and logs for every experiment.
        ├── src                                        - contains project source code.
        │   │
        │   ├── configs                                - contains configurable parameters, so you can change them in the future.
        │   │     ├── config.py                        - contains YAML config parser.
        │   │     └── config.yaml                      - contains training hyperparameters, folder paths, models architecture, params for optimizers and lr_schedulers.
        │   │                          
        │   ├── models                                 - contains all used models.
        │   │     │
        │   │     ├── unet                             - contains UNet model files.
        │   │     │     ├── unet_model.py              - contains implemented UNet model architecture.
        │   │     │     ├── unet_trainer.py            - contains Pytorch Lightning Trainer for UNet model.
        │   │     │     └── unet_inference.py          - contains script to use trained UNet model to make preditions on raw data input.
        │   │     │
        │   │     ├── fast_scnn                        - contains Fast SCNN model files.
        │   │     │       ├── fast_scnn_model.py       - contains implemented Fast SCNN model architecture.
        │   │     │       ├── fast_scnn_trainer.py     - contains Fast SCNN Pytorch Lightning Trainer.
        │   │     │       └── fast_scnn_inference.py   - contains script to use trained Fast SCNN model to make preditions on raw data input. 
        │   │     │
        │   │     └── mask_rcnn                        - contains Mask R-CNN model files.
        │   │             ├── mask_rcnn_model.py       - contains implemented Mask R-CNN model architecture.
        │   │             ├── mask_rcnn_trainer.py     - contains script to create the Mask R-CNN model.
        │   │             └── mask_rcnn_inference.py   - contains script to use trained Mask RCNN model to make preditions on raw data input. 
        │   │
        │   │
        │   ├── processing                             - contains processing data files.
        │   │       ├── dataset.py                     - contains the scpript to store the data samples and their corresponding target annotations.
        │   │       ├── transforms.py                  - contains the scpript for data augmentations and transformations.
        │   │       └── datamodule.py                  - contains Pytorch Lightning datamodule that encapsulates all the steps needed to process data.
        │   │ 
        │   │                                
        │   └── utils                                  - contains utilities functions.
        │         └── utils.py                         - contains visualizing, metrics, and commonly used helper functions.
        │
        │
        ├── tests                                      - contains pytest tests.
        │    │
        │    ├── models                                - contains segmentation models tests.
        │    │     │
        │    │     ├── unet                            - contains pytest tests for UNet model and trainer.
        │    │     │    ├── unet_model_test.py         - contains the scpript for UNet model tests.
        │    │     │    └── unet_trainer_test.py       - contains the scpript for UNet trainer tests.
        │    │     │
        │    │     ├── fast_scnn                       - contains pytest tests for Fast SCNN model and trainer.
        │    │     │    ├── fast_scnn_model_test.py    - contains the scpript for Fast SCNN model tests.
        │    │     │    └── fast_scnn_trainer_test.py  - contains the scpript for Fast SCNN trainer tests.
        │    │     │
        │    │     └── mask_rcnn                       - contains pytest tests for Mask R-CNN model and trainer.
        │    │          ├── mask_rcnn_model_test.py    - contains the scpript for Mask R-CNN model tests.
        │    │          └── mask_rcnn_trainer_test.py  - contains the scpript for Mask R-CNN trainer tests.
        │    │
        │    └── processing                            - contains processing data tests.
        │           ├── dataset_test.py                - contains the scpript for Dataset tests.
        │           ├── transforms_test.py             - contains the scpript for data transforms and augmentations tests.
        │           └── datamodule_test.py             - contains the scpript for Datamodule tests.
        │
        ├── requirements.txt                           - contains required python packages to run this project.
        └── train.py                                   - contains train model script.
```

## Setup Environment 

Replace tests, utils and config files with updated ones for new models:   
1. [tests](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/code/tests.zip) for `./tests`
2. [utils](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/code/utils.py.zip) for `./src/utils`
3. [config.yaml](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/code/config.yaml.zip) for `./src/configs`

Now it's time to implement new models and add them to the project!

## Fast SCNN

So the first model will be the [Fast SCNN](https://arxiv.org/abs/1902.04502) ── semantic segmentation model for high-resolution image data. This model is one of the best ways to do segmentations efficiently. The model shares the computations between two encoders to build a fast semantic segmentation network:

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/fast-scnn.png">
</div>

### Defining Fast SCNN model

#### ConvBNReLU Block

One of the most straightforward blocks of the Fast SCNN architecture is the `ConvBNReLU` block. It contains only `Conv2d` layers followed by the `BatchNorm` layer and `RelU` activation function. For the convolution layer, use a `3x3` kernel size with a `stride=1` and `padding=0`. For `ReLU` activation, use `inplace=True`.

Implement the following:

* `nn.Sequential` should contain `Conv2d(in_channels, out_channels) -> BatchNorm2d -> ReLU`.
* `bias=False`.
* Use `padding=0` and `stride=1`.
* Use `inplace=True` for `ReLU`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """
    Conv-BatchNorm-ReLU block.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param kernel_size: a size of the conv kernel.
    :param stride: a stride of the conv.
    :param padding: padding added to all four sides of the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the ConvBNReLU block.

        :param x: an input with in_channels.
        :return: an output with out_channels.
        """
        out = ...
        return out

                . . .
```

Test your `ConvBNReLU` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_conv_bn_relu` function.

#### DSConv Block

This block is a [depthwise separable convolution](https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/) block with two layers. The `DSConv` block is computationally more efficient than usual depthwise convolution. It contains two `Conv2d` layers followed by `BatchNorm` layers and `RelU` activation functions. For the first convolutional layer, use a `3x3` kernel size with a `stride=1` and `padding=1`; for the second layer, use a `1x1` kernel with `stride=1`. For `ReLU` activations, use `inplace=True`.

Implement the following:

* `nn.Sequential` should contain `Conv2d(dw_channels, dw_channels) -> BatchNorm2d -> ReLU -> Conv2d(dw_channels, out_channels) -> BatchNorm2d -> ReLU`.
* Use `padding=1`, `stride=1` and `groups=dw_channels` for the first Conv2d.
* Use `stride=1` for the second Conv2d.
* Use `bias=False` for both Conv2d.
* Use `inplace=True` for `ReLU`.

```python
                . . .

class DSConv(nn.Module):
    """
    Depthwise Separable Convolutions.

    :param dw_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, out_channels, stride=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the DSConv block.

        :param x: an input with dw_channels.
        :return: an output with out_channels.
        """
        out = ...
        return out

                . . .
```

Test your `DSConv` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_ds_conv` function.

#### DWConv Block

This block is a [depthwise convolution](https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/) block with one layer. It contains a `Conv2d` layer followed by a `BatchNorm` layer and `RelU` activation function. The convolutional layer uses a `3x3` kernel size with a `stride=1` and `padding=1` and groups to control the connections between inputs and outputs. For `ReLU` activations, use `inplace=True`.

Implement the following:

* `nn.Sequential` should contain `Conv2d(dw_channels, dw_channels) -> BatchNorm2d -> ReLU`.
* Use `padding=1`, `stride=1` and `groups=dw_channels`.
* Use `bias=False`.
* Use `inplace=True` for `ReLU`.

```python
                . . .

class DWConv(nn.Module):
    """
    Depthwise Convolutions.

    :param dw_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, out_channels, stride=1):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the DWConv block.

        :param x: an input with dw_channels.
        :return: an output with out_channels.
        """
        out = ...
        return out

                . . .
```

Test your `DWConv` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_dw_conv` function.

#### Linear Bottleneck

Fast SCNN relies on depthwise separable convolutions and [linear bottleneck blocks](https://towardsdatascience.com/residual-bottleneck-inverted-residual-linear-bottleneck-mbconv-explained-89d7b7e7c6bc) to reduce the number of params and matrix multiplications. It contains four layers: `ConvBNReLU`, `DWConv`, `Conv2d`, and `BatchNorm`. Sometimes residual hasn't had the same output dimension, so we cannot add them. In this case, apply shortcuts if the input and output features are different.

Implement the following:

* Use shortcut if `stride == 1` and `in_channels == out_channels`.
* `nn.Sequential` should contain `ConvBNReLU(in_channels, in_channels * t, kernel_size=1) -> DWConv(in_channels * t, in_channels * t, stride) -> Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False) -> BatchNorm2d`.

```python
                . . .

class LinearBottleneck(nn.Module):
    """
    Linear Bottleneck.
    Allows to enable high accuracy and performance.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    :param t: an expansion factor.
    :param stride: a stride of the conv.
    """
    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super(LinearBottleneck, self).__init__()
        # use residual add if features match, otherwise a normal Sequential
        self.use_shortcut = ...
        self.block = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the LinearBottleneck block.

        :param x: an input with in_channels.
        :return: an output with out_channels.
        """
        # Use conv block
        out = self.block(x)
        # Check use_shortcut or not
        if self.use_shortcut:
            out = x + out
        return out

                . . .
```

Test your `LinearBottleneck` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_linear_bottleneck` function.

#### Pyramid Pooling

This module contains information with different scales and varies among different subregions. The pyramid module has four levels of pooling kernels. Each of these kernels gathers different levels of contextual information. The pooling kernels use the average pooling operation. The kernel sizes are `1x1`, `2x2`, `3x3`, and `6x6`.

The sequence of operations of the Pyramid Pooling Module is as follows: The PPM takes the feature maps from the Convolutional layer and then applies average pooling and upscaling functions to harvest different subregion representations, and then they are concatenated together. It helps carry local and global context information from the image making the segmentation process more accurate.

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/ppm.png">
</div>


Implement the following:

* In `__init__` method:
    * Get `inter_channels` as floored 1/4 of the `in_channels`.
    * Create four `ConvBNReLU` layers with `in_channels`, `inter_channels` and `kernel_size=1` inputs.
    * Create final `ConvBNReLU` layer with `in_channels * 2`, `out_channels` and `kernel_size=1` input.
* In `forward` method:
    * Find `x` input [H,W] `size`.
    * For `feat1` do the following: pool `x` with kernel `1`, apply `conv1` and unsample with `size`.
    * For `feat2` do the following: pool `x` with kernel `2`, apply `conv1` and unsample with `size`.
    * For `feat3` do the following: pool `x` with kernel `3`, apply `conv1` and unsample with `size`.
    * For `feat4` do the following: pool `x` with kernel `6`, apply `conv1` and unsample with `size`.
    * Concatenate `x` with `feats`.
    * Apply the final conv layer.

```python
                . . .

class PyramidPooling(nn.Module):
    """
    Pyramid pooling module.
    Aggregates the different-region-based context information.

    :param in_channels: a number of channels in the input.
    :param out_channels: a number of channels produced by the conv.
    """
    def __init__(self, in_channels, out_channels):
        super(PyramidPooling, self).__init__()
        inter_channels = ...
        self.conv1 = ...
        self.conv2 = ...
        self.conv3 = ...
        self.conv4 = ...
        self.out = ...

    @staticmethod
    def upsample(x, size):
        """
        Up samples the input.

        :param x: an input.
        :param size: a size to up sample the input.
        :return: an up sampled input.
        """
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    @staticmethod
    def pool(x, size):
        """
        Applies a 2D adaptive average pooling over an input.

        :param x: an input.
        :param size: the target output size.
        :return: a pooled input.
        """
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        """
        Forward pass through the PyramidPooling block.

        :param x: an input.
        :return: an output of the PPM.
        """
        size = ...
        feat1 = ...
        feat2 = ...
        feat3 = ...
        feat4 = ...
        x = ...
        x = ...
        return x

                . . .
```

Test your `PyramidPooling` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_pyramid_pooling` function.

#### Learning To Downsample

Learning to Downsample module uses three layers to extract global features. It contains three layers: a `Conv2D` layer followed by two Depthwise Separable Convolutional Layers. Here, all three layers use a stride of `2`. The `ConvBNReLU` layer uses a kernel size of `3x3`.

Implement the following:
* Use `ConvBNReLU` for the first conv layer with `in_channels=3`, `out_channels=dw_channels1`, `kernel_size=3` and `stride=2`.
* Use Depthwise Separable Convolutional Layer for `dsconv1` with `in_channels=dw_channels1`, `out_channels=dw_channels2` and `stride=2`.
* Use Depthwise Separable Convolutional Layer for `dsconv2` with `in_channels=dw_channels2`, `out_channels=out_channels` and `stride=2`.

```python
                . . .

class LearningToDownsample(nn.Module):
    """
    Learning to downsample module.

    :param dw_channels1: a number of channels to downsample.
    :param dw_channels2: a number of channels to downsample.
    :param out_channels: a number of channels produced by the conv.
    """
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.conv = ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        """
        Forward pass through the LearningToDownsample block.

        :param x: an input.
        :return: an output of the LTD.
        """
        # Apply conv
        x = ...
        # Apply dsconv1
        x = ...
        # Apply dsconv2
        x = ...
        return x

                . . .
```

Test your `LearningToDownsample` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_learning_to_downsample` function.

#### Global Feature Extractor

The global feature extractor module captures the global context for the segmentation. It directly takes the output from the Learning to Downsample module. It contains three Linear Bottleneck blocks followed by a PyramidPooling module.

Implement the following:
* `bottleneck1`: use `_make_layer` method to create `nn.Sequential` of three `LinearBottleneck` blocks with `inplanes=in_channels`, `planes=block_channels[0]`, expansion factor `t` and `stride=2`.
* `bottleneck2`: use `_make_layer` method to create `nn.Sequential` of three `LinearBottleneck` blocks with `inplanes=block_channels[0]`, `planes=block_channels[1]`, expansion factor `t` and `stride=2`.
* `bottleneck3`: use `_make_layer` method to create `nn.Sequential` of three `LinearBottleneck` blocks with `inplanes=block_channels[1]`, `planes=block_channels[2]`, expansion factor `t` and `stride=1`.
* Create a pyramid pooling module.

```python
                . . .

class GlobalFeatureExtractor(nn.Module):
    """
    Global feature extractor module.

    :param in_channels: number of channels in the input.
    :param block_channels: list with number of channels produced by the LinearBottleneck.
    :param out_channels: number of output channels.
    :param t: an expansion factor.
    :param num_blocks: number of times block is repeated.
    """
    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = ...
        self.bottleneck2 = ...
        self.bottleneck3 = ...
        self.ppm = ...

    @staticmethod
    def _make_layer(block, inplanes, planes, blocks, t=6, stride=1):
        """

        :param block: block to create.
        :param inplanes: number of input channels.
        :param planes: number of output channels.
        :param blocks: number of times block is repeated.
        :param t: an expansion factor.
        :param stride: a stride of the conv.
        :return: nn.Sequential of layers.
        """
        layers = [block(inplanes, planes, t, stride)]
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the GlobalFeatureExtractor block.

        :param x: an input.
        :return: an output of the GFE.
        """
        x = ...
        x = ...
        x = ...
        x = ...
        return x

                . . .
```

Test your `GlobalFeatureExtractor` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_global_feature_extractor` function.

#### Feature Fusion Module

In the Feature Fusion Module module, two inputs are added together. The first one is the output from the Learning to Downsample module. The output from this Learning to Downsample module is pointwise convoluted at first before adding to the second input.

The Second input is the output from the Global Feature Extractor. But before adding the second input, they first Upsampled by the factor of `(4,4)`, depthWise convoluted, and finally followed by another pointwise convolution. After adding these two inputs, we use the `ReLU` activation function.

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/fmm.png">
</div>


Implement the following:
* for DepthWise convolution use `DWConv` block with `in_channels=lower_in_channels`.
* for lower resolution input create pointwise convolution using `nn.Sequential`, which should contain `Conv2d(out_channels, out_channels, kernel_size=1) -> BatchNorm2d`.
* for higher resolution input create pointwise convolution using `nn.Sequential`, which should contain `Conv2d(highter_in_channels, out_channels, kernel_size=1) -> BatchNorm2d`.
* Use `inplace=True` for `ReLU`.

```python
                . . .

class FeatureFusionModule(nn.Module):
    """
    Feature fusion module.

    :param highter_in_channels: high resolution channels input.
    :param lower_in_channels:  low resolution channels input.
    :param out_channels: number of output channels.
    :param scale_factor: scale factor.
    """
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = ...
        self.dwconv = ...
        self.conv_lower_res = nn.Sequential(...)
        self.conv_higher_res = nn.Sequential(...)
        self.relu = ...

    def forward(self, higher_res_feature, lower_res_feature):
        """
        Forward pass through the FeatureFusionModule block.

        :param higher_res_feature: high resolution features.
        :param lower_res_feature: low resolution features.
        :return: an output of the FFM.
        """
        # Upsample lower_res_feature using F.interpolate
        # with args scale_factor=4, mode='bilinear', align_corners=True
        lower_res_feature = ...
        # Apply dwconv
        lower_res_feature = ...
        # Apply conv_lower_res
        lower_res_feature = ...

        # Apply conv_higher_res on higher_res_feature
        higher_res_feature = ...
        # Add convs outputs and apply ReLU
        out = ...
        return out

                . . .
```

Test your `FeatureFusionModule` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_feature_fusion_module` function.

#### Classifier

In Classifier, two Depthwise Separable Convolutions layers are followed by one Pointwise Convolutional layer.

Implement the following:
* Create two Depthwise Separable Convolutions with `dw_channels=dw_channels`, `out_channels=dw_channels` and `stride`.
* Create Pointwise Convolution using `nn.Sequential`, which should contain `Dropout(0.1) -> Conv2d(dw_channels, num_classes, kernel_size=1)`.

```python
                . . .
class Classifier(nn.Module):
    """
    Classifier.

    :param dw_channels: number of channels for dsconv.
    :param num_classes: number of classes.
    :param stride: a stride of the conv.
    """
    def __init__(self, dw_channels, num_classes, stride=1):
        super(Classifier, self).__init__()
        self.dsconv1 = ...
        self.dsconv2 = ...
        self.conv = ...

    def forward(self, x):
        """
        Forward pass through the Classifier block.

        :param x: an input.
        :return: an output of the Classifier.
        """
        x = ...
        x = ...
        x = ...
        return x

                . . .
```

Test your `Classifier` class in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_classifer` function.

#### Put it all together

To complete the Fast RCNN architecture, use four main blocks: Learning To Downsample, Global Feature Extractor, Feature Fusion Module, and Classifier.

Implement the following:
* Create the Learning To Downsample module with `32, 48, 64` number of channels.
* Create the Global Feature Extractor module with `in_channels=64`, `block_channels=[64, 96, 128]`, `out_channels=128`, `t=6` and `num_blocks=[3, 3, 3]`.
* Create the Feature Fusion Module module with `highter_in_channels=64`, `lower_in_channels=128`, `out_channels=128`.
* Create the Classifier module with `dw_channels=128` and `num_classes`.

```python
                . . .
class FastSCNN(nn.Module):
    """
    The complete architecture of FastSCNN using layers defined above.

    :param num_classes: number of classes.
    """
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = ...
        self.global_feature_extractor = ...
        self.feature_fusion = ...
        self.classifier = ...

    def forward(self, x):
        """
        Forward pass through the FastSCNN.

        :param x: an input.
        :return: an output of the FastSCNN.
        """
        # Get input size [H, W]
        size = ...
        # Apply Learning To Downsample
        higher_res_features = ...
        # Apply Global Feature Extractor
        x = ...
        # Apply Feature Fusion
        x = ...
        # Apply Classifier
        x = ...
        # Upsample with size
        # Use F.interpolate with params 
        # mode='bilinear' and align_corners=True
        out = ...
        return outs

```

Test your full FastSCNN model architecture in the `tests/models/fast_scnn/fast_scnn_model_test.py` file running the `test_full_fast_scnn` function.


### Defining Pytorch Lightning Fast SCNN trainer

Now let's start Defining the Pytorch Lightning Fast SCNN trainer in the `src/models/fast_scnn/fast_scnn_trainer.py` file. As a loss function use the [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss). The steps will be very similar to those that you did for Unet model.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer, Adam
from typing import Dict, Tuple, Union, List
from torch.optim.lr_scheduler import StepLR

from src.utils.utils import masks_iou
from src.configs.config import CONFIG
from src.models.fast_scnn.fast_scnn_model import FastSCNN
from src.utils.utils import get_batch_images_and_pred_masks_in_a_grid


class FastSCNNTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the FastSCNN model.

    :param n_classes: number of classes to predict.
    """
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.model = FastSCNN(n_classes)
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                      batch_idx: int) -> torch.Tensor:
        pass
    
    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        pass
    
    def training_epoch_end(self, train_outputs: List[Dict[str, torch.Tensor]]) -> None:
        pass
    
    def validation_epoch_end(self, val_outputs: List[Dict[str, torch.Tensor]]) -> None:
        pass
    
    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        pass
```

#### Training Step

```python
                . . .

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                      batch_idx: int) -> torch.Tensor:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: step loss.
        """
        images, targets = batch

        images_tensor = ...
        masks_tensor = ...

        y_hat = ...
        loss = ...

        return loss                

                . . .
```

Test your `training_step()` method in the `tests/models/fast_scnn/fast_scnn_trainer_test.py` file running the `test_training_step` function.

#### Validation Step

```python
                . . .

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Take a batch from the validation dataset and input its images into the model.
        Retrieves losses after one validation step and mask IoU score.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: dict with epoch loss value and masks IoU score and 
        predicted masks for one batch step.
        """
        images, targets = batch

        images_tensor = ...
        masks_tensor = ...

        y_hat = ...

        masks_iou_score = ...
        loss = ...
        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(y_hat, images)

        return {'val_loss': loss, 'val_iou': masks_iou_score, 'val_images_and_pred_masks': imgs_grid}

                . . .
```

Test your `validation_step()` method in the `tests/models/fast_scnn/fast_scnn_trainer_test.py` file running the `test_validation_step` function.

#### Epochs logging and configure optimizers

Just paste the following code into the `src/models/mask_rcnn/fast_scnn_trainer.py` file:

```python
    def training_epoch_end(self, train_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        Calculates and logs mean total loss at the end
        of the training epoch with the losses of all training steps.

        :param train_outputs: dicts with losses of all training steps.
        :return: a dict with losses and predicted mask IoU for one batch step.
        """
        loss_epoch = torch.stack([output['loss'] for output in train_outputs]).mean()

        self.log('train/loss_epoch', loss_epoch.item())
        
    def validation_epoch_end(self, val_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        Calculates and logs mean total loss and average IoU over predicted masks at the end of the
        validation epoch with the outputs of all validation steps.

        :param val_outputs: losses and predicted masks IoU of all validation steps.
        :return: epoch loss and average masks IoU score values.
        """
        loss_epoch = torch.stack([output['val_loss'] for output in val_outputs]).mean()
        avg_masks_iou = torch.stack([output['val_iou'] for output in val_outputs]).mean()

        self.log('val/loss_epoch', loss_epoch.item(), prog_bar=True)
        self.log('val/val_iou', avg_masks_iou.item(), prog_bar=True)

        # log predicted masks for validation dataset
        for ind, dict_i in enumerate(val_outputs):
            self.logger.experiment.add_image('Predicted masks on images', dict_i['val_images_and_pred_masks'],
                                             ind)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        """
        Configure the Adam optimizer and the StepLR learning rate scheduler.

        :return: a dict with the optimizer and lr_scheduler.
        """
        optimizer = Adam(self.model.parameters(), lr=CONFIG['fast_scnn']['optimizer']['initial_lr'].get())
        lr_scheduler = StepLR(optimizer,
                              step_size=CONFIG['fast_scnn']['lr_scheduler']['step_size'].get(),
                              gamma=CONFIG['fast_scnn']['lr_scheduler']['gamma'].get())

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
```


### Defining Fast SCNN Inference

To run the Fast SCNN model with the raw data past the following code block into `src/models/fast_scnn/fast_scnn_inference.py`:

```python
import torch
import numpy as np
from torchvision import transforms as T

from src.configs.config import CONFIG
from src.models.fast_scnn.fast_scnn_model import FastSCNN


def model_inference(trained_model_path: str, image: torch.Tensor) -> np.ndarray:
    """
    Loads the entire trained Fast SCNN model and predicts segmentation mask for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted mask for an input image.
    """

    model = FastSCNN(CONFIG['fast_scnn']['model']['n_classes'].get())
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    resize_to = (
        CONFIG['data_augmentation']['resize_to'].get(),
        CONFIG['data_augmentation']['resize_to'].get()
    )

    resized_image = T.Resize(resize_to)(image)

    with torch.no_grad():
        prediction = model(resized_image)
        mask = torch.sigmoid(prediction[0])

    orig_size = (image.shape[2], image.shape[3])
    resized_mask_to_original_image_size = T.Resize(orig_size)(mask)

    return resized_mask_to_original_image_size.numpy()
```

## Mask R-CNN

The second model is the Mask R-CNN.

The R-CNN approach utilises bounding boxes across the object regions, which then evaluates convolutional networks independently on all the Regions of Interest (ROI) to classify multiple image regions into the proposed class.
The R-CNN architecture was improved into Faster R-CNN, and, in turn, Faster R-CNN is a base of Mask R-CNN.

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/rcnn_region_based_convolutional_network.png">
</div>


["Faster" R-CNN](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection) has two stages:

1. **Region Proposal Network (RPN)**. [RPN](https://medium.com/egen/region-proposal-network-rpn-backbone-of-faster-r-cnn-4a744a38d7f9#:~:text=Region%20Proposal%20Network%20%28RPN%29%20—%20Backbone%20of%20Faster%20R%2DCNN,-Tanay%20Karmarkar&text=In%20object%20detection%20using%20R,identifiable%20within%20a%20particular%20image.) is simply a Neural Network that proposes multiple objects available within a particular image.

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/rpn.png">
</div>


2. **Fast R-CNN**. It extracts features using RoIPool (Region of Interest Pooling) from each candidate box and performs classification and bounding-box regression. RoIPool is an operation for extracting a small feature map from each RoI in detection.

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/fast-rcnn.png">
</div>


As its name suggests, Faster R-CNN is faster than Fast R-CNN thanks to the region proposal network (RPN). The convolution operation is done only once per image to generate a feature map, so you don’t have to feed 2’000 region proposals to the CNN every time.

The Mask R-CNN loss function in train mode is calculated as follows:

$$Loss = Loss_{cls} + Loss_{box} + Loss_{mask}$$

Where: 
* $Loss$ represents the total cost loss function; 
* $Loss_{cls}$ represents the classification loss of the prediction box;  
* $Loss_{box}$ represents the regression loss of the prediction box; 
* $Loss_{mask}$ represents the average binary cross-entropy loss.

Faster R-CNN has two outputs for each candidate object, a class label and a bounding-box offset, and Mask R-CNN is the addition of a third branch that outputs the object mask (Region of Interest). 

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p2/img/architecture_of_mask_rcnn.png">
</div>

### Defining Mask R-CNN model

We will use a pre-trained model and just finetune the last layer, given that our dataset is pretty small.
This will prepare the model to be trained and evaluated on our dataset.

Now your task is to implement a `create_model` function for Mask-RCNN model:

```python
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.configs.config import CONFIG


def create_model(n_classes: int, pretrained: bool = True) -> MaskRCNNPredictor:
    """
    Creates the Mask R-CNN model based on `maskrcnn_resnet50_fpn`.

    :param n_classes: number of classes the model should predict.
    :param pretrained: an indicator to load pretrained model or not.
    :return: MaskRCNNPredictor.
    """
    # Loading an instance of segmentation model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Getting the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one,
    # use Fast RCNN predictor with args: the number of input features
    # and the number of classes
    model.roi_heads.box_predictor = ...

    # Getting the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = CONFIG['mask_rcnn']['model']['hidden_size'].get()
    # Replace the mask predictor with a new one,
    # using Mask-RCNN predictor with args: the number of 
    # input features for the mask classifier,
    # the number of hidden layers and the number of classes  
    model.roi_heads.mask_predictor = ...

    return model
``` 
Test your `create_model()` function in the `tests/models/mask_rcnn/mask_rcnn_model_test.py` file.


### Defining Pytorch Lightning Mask R-CNN trainer

After defining the model, let's move on to implement our Pytorch Lighting module in the `src/models/mask_rcnn/mask_rcnn_trainer.py` file:

```python
import torch
import pytorch_lightning as pl
from typing import Dict, Tuple, Union
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

from src.configs.config import CONFIG
from src.utils.utils import bboxes_iou
from src.models.mask_rcnn.mask_rcnn_model import create_model
from src.utils.utils import get_batch_images_and_pred_masks_in_a_grid

accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'


class MaskRCNNTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the torchvision Mask R-CNN model.

    :param n_classes: number of classes of the Mask R-CNN (including the background).
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.model = create_model(n_classes=n_classes)

    @staticmethod
    def _convert_targets_to_mask_rcnn_format(targets: Tuple[Dict[str, torch.Tensor]]
                                             ) -> Tuple[Dict[str, torch.Tensor]]:
        pass

    def step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        pass

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                      batch_idx: int) -> Dict[str, torch.Tensor]:
        pass

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                        batch_idx: int) -> Dict[str, Union[torch.Tensor, dict]]:
        pass

    def training_epoch_end(self, train_outputs: Tuple[Dict[str, torch.Tensor], ...]) -> None:
        pass

    def validation_epoch_end(self, val_outputs: Tuple[Dict[str, Union[torch.Tensor, 
                             Dict[str, torch.Tensor]]], ...]) -> None:
        pass

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        pass

```

#### Step

Implement a shared `step` method for training and validation scheme:
```python
                . . .

    @staticmethod
    def _convert_targets_to_mask_rcnn_format(targets: Tuple[Dict[str, torch.Tensor]]
                                             ) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Converts targets from datamodule batch to Mask-RCNN format.
        :param targets: targets from datamodule batch.
        """
        mask_rcnn_targets = []

        # For Mask RCNN training, you need your Dataset to consist of `masks`, `boxes`, `labels` and `image_id` keys.
        # So, your task here is to apply In-place modification of your Dataset dictionary.
        for target in targets:
            # Get bounding box coordinates for car instance on the mask.
            # Bounding box format is [x0, y0, x1, y1],
            # where (x0, y0) are coordinates of upper left corner and 
            # (x1, y1) are coordinates of lower right corner of the bounding box.
            mask = target['mask']

             . . . 

            # Convert a bbox into torch.float32 data type
            box = ...

            mask_rcnn_target = {
                'image_id': target['image_id'],
                'boxes': box.to(accelerator),
                'masks': torch.as_tensor(target['mask'], dtype=torch.uint8),
                'labels': torch.tensor([1], dtype=torch.int64).to(accelerator),
            }
            mask_rcnn_targets.append(mask_rcnn_target)

        return tuple(mask_rcnn_targets)

    def step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        A shared step for the training_step and validation_step functions.
        Calculates the sum of losses for one batch step.
        :param batch: a batch of images and targets with annotations.
        :return: a dict of the losses sum and loss mask of the prediction heads per one batch step.
        """
        self.model.train()

        images, targets = batch
        
        # Convert targets to Mask RCNN format
        targets = ...
        # Put images and targets to the model
        outputs = ...
        # Outputs is a dict which contains train losses.
        # Calculate the sum of these losses
        loss_step = ...
        # Take a `loss_mask` from outputs dict
        loss_mask = ...

        return {'loss': loss_step, 'loss_step': loss_step.detach(), 'loss_mask': loss_mask.detach()}

                . . .

```

Test your `_convert_targets_to_mask_rcnn_format()` and `step()` methods in the `tests/models/mask_rcnn/mask_rcnn_trainer_test.py` file running the `test_convert_targets_to_mask_rcnn_format` and `test_step` functions respectively.


#### Training Step

Implement `training_step` method for training the model:
```python
                . . .

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                      batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.
        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict of the losses sum and loss_mask of the prediction heads for one batch step.
        """
        # Use a shared step method
        outputs = ...

        self.log('train/loss_step', outputs['loss_step'].item())
        self.log('train/loss_mask_step', outputs['loss_mask'].item())

        return outputs

                . . .

```

Test your `training_step()` method in the `tests/models/mask_rcnn/mask_rcnn_trainer_test.py` file running the `test_training_step` function.

#### Validation Step

Implement `validation_step` method for model validation:
```python
                . . .

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor]]],
                        batch_idx: int) -> Dict[str, Union[torch.Tensor, dict]]:
        """
        Take a batch from the validation dataset and input its images into the model.
        Retrieves losses, predicted masks and IoU metric after one validation step.
        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict of the validation step losses of the prediction heads,
        intersection over union metric score and predicted masks for one batch step.
        """
        # Use a shared step on the batch.
        # (We use here a step method to get loss 
        # for one validation step)
        outputs = ...

        self.model.eval()
        images, targets = batch

        # Put images as input to the self.model
        # to retrieve the predicted masks and boxes
        preds = ...

        # Convert targets to Mask RCNN format
        targets = ...
        bboxes_iou_score = torch.stack([bboxes_iou(t, o) for t, o in zip(targets, preds)]).mean()
        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(outs, images, mask_rcnn=True)

        return {'val_outputs': outputs, 'val_images_and_pred_masks': imgs_grid, 'val_iou': bboxes_iou_score}

                . . .

```

Test your `validation_step()` method in the `tests/models/mask_rcnn/mask_rcnn_trainer_test.py` file running the `test_validation_step` function.


#### Epochs logging and configure optimizers

Just put the rest of the code in the `src/models/mask_rcnn/mask_rcnn_trainer.py` file:

```python
                . . .

    def training_epoch_end(self, train_outputs: Tuple[Dict[str, torch.Tensor], ...]) -> None:
        """
        Calculates and logs mean total loss and loss_mask
        at the end of the training epoch with the outputs of all training steps.
        :param train_outputs: outputs of all training steps.
        :return: mean loss and loss_mask for one training epoch.
        """
        loss_epoch = torch.hstack([outputs['loss_step'] for outputs in train_outputs]).mean()
        loss_mask_epoch = torch.hstack([outputs['loss_mask'] for outputs in train_outputs]).mean()

        self.log('train/loss_epoch', loss_epoch.item())
        self.log('train/loss_mask_epoch', loss_mask_epoch.item())

    def validation_epoch_end(self, val_outputs: Tuple[Dict[str, Union[torch.Tensor, 
                             Dict[str, torch.Tensor]]], ...]) -> None:
        """
        Calculates and logs mean total loss, loss_mask and IoU metric
        at the end of the validation epoch with the outputs of all validation steps.
        :param val_outputs: outputs of all validation steps.
        :return: mean loss, loss_mask and average boxes IoU score for one training epoch.
        """
        loss_epoch = torch.hstack([dict_i['loss_step'] for dict_i in
                                   [outs['val_outputs'] for outs in val_outputs]]).mean()
        loss_mask_epoch = torch.hstack([dict_i['loss_mask'] for dict_i in
                                        [outs['val_outputs'] for outs in val_outputs]]).mean()
        avg_iou = torch.hstack([val_iou for val_iou in [outs['val_iou'] for outs in val_outputs]]).mean()

        self.log('val/loss_epoch', loss_epoch.item(), prog_bar=True)
        self.log('val/loss_mask_epoch', loss_mask_epoch.item())
        self.log('val/val_iou', avg_iou.item(), prog_bar=True)
        #  log predicted masks for validation dataset
        for ind, dict_i in enumerate(val_outputs):
            self.logger.experiment.add_image('Predicted masks on images', dict_i['val_images_and_pred_masks'], ind)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        """
        Configure the SGD optimizer and the StepLR learning rate scheduler.
        :return: a dict with the optimizer and lr_scheduler.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = SGD(params, lr=CONFIG['mask_rcnn']['optimizer']['initial_lr'].get(),
                        momentum=CONFIG['mask_rcnn']['optimizer']['momentum'].get(),
                        weight_decay=CONFIG['mask_rcnn']['optimizer']['weight_decay'].get())
        lr_scheduler = StepLR(optimizer, step_size=CONFIG['mask_rcnn']['lr_scheduler']['step_size'].get(),
                              gamma=CONFIG['mask_rcnn']['lr_scheduler']['gamma'].get())
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
```

### Defining Mask R-CNN Inference

To run the Mask R-CNN model with the raw data past the following code block into `src/models/mask_rcnn/mask_rcnn_inference.py`:

```python
import torch
import numpy as np
from typing import Tuple

from src.configs.config import CONFIG
from src.models.mask_rcnn.mask_rcnn_model import create_model


def model_inference(trained_model_path: str, image: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """
    Loads the entire trained Mask R CNN model and predicts boxes, labels, scores and
    segmentation masks for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted numpy boxes, labels, scores, masks for an input image.
    """

    model = create_model(n_classes=CONFIG['mask_rcnn']['model']['n_classes'].get(), pretrained=False)
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    with torch.no_grad():
        predictions = model(image)

    boxes, labels, scores, masks = predictions[0]['boxes'].numpy(),\
                                   predictions[0]['labels'].numpy(), \
                                   predictions[0]['scores'].numpy(), \
                                   predictions[0]['masks'].numpy()

    return boxes, labels, scores, masks
```

Now you have finished defining our two new models and their Pytorch Lightning trainers. Next, you will do training them on dataset and perform inference on raw data.

## Models Training and Inference

### Models Training

To train new model use the same training function, that you use for UNet training, from `./train.py`. Training models with Google Colab GPU:

1. Compress your project
2. Upload it on your `Google Drive`
3. Go to `Google Colab` and create a new notebook
4. Set the runtime accelerator as `GPU`: `Runtime -> Change runtime type -> GPU`
5. Mount Google Drive where you store your project:
```python
from google.colab import drive
drive.mount('/content/drive')
```
6. Move to your project archive:
```python
% cd path/to/your/project
```
7. Unzip your project:
```python
!unzip DLProject.zip
```
8. Move to your project root:
```python
% cd DLProject
```
9. Install all the dependencies:
```python
! pip install -r requirements.txt
```

If you face some `PIL` import or attribute errors just restart your notebook runtime and run the cells from steps 5, 6 and 8.

When requirements are installed use: 

* Fast SCNN:
```python
from train import train_model
from src.models.fast_scnn.fast_scnn_trainer import FastSCNNTrainer

model = FastSCNNTrainer()
train_model(exp_number=2, model=model, batch_size=16, max_epochs=3, use_resize=False, use_random_crop=True, use_hflip=True)
```

* Mask R-CNN:
```python
from train import train_model
from src.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer

model = MaskRCNNTrainer()
train_model(exp_number=3, model=model, batch_size=4, max_epochs=1, use_resize=False, use_random_crop=False, use_hflip=True)
```

Logs checking in TensorBoard:
```python
%reload_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/path/to/your/project/DLProject/experiments
```

### Models Inference

During training each model, you will save a model checkpoint for every epoch and a fully trained model after the entire train in the `experiments/exp_{exp_number}/weights` folder. So let's check the model predictions on a raw data using `model_inference` function from `src/models/fast_scnn/fast_scnn_inference.py` and `src/models/mask_rcnn/mask_rcnn_inference.py`

* Fast SCNN:
```python
from src.utils.utils import get_input_image_for_inference
from src.models.fast_scnn.fast_scnn_inference import model_inference
from src.utils.utils import show_pic_and_pred_semantic_mask

# Provide a path to model.pt checkpoint of your trained model
trained_model_path = 'path/to/your/project/DLProject/experiments/exp_{exp_number}/weights/model.pt'

# You may check your model performance with one of the image from dataset 
# or any other car image on your Google Drive
path_image = 'path/to/local/image'

# Or you also may check it with the any image in the internet
# url_to_image = 'url/to/internet/image'

image = get_input_image_for_inference(local_path=path_to_local_image) 
# image = get_input_image_for_inference(url=url_to_image) 

mask = model_inference(trained_model_path, image)

# Visualizing results
show_pic_and_pred_semantic_mask(image, mask)
```

* Mask R-CNN:
```python
from src.utils.utils import get_input_image_for_inference
from src.models.mask_rcnn.mask_rcnn_inference import model_inference
from src.utils.utils import show_pic_and_pred_instance_masks

# Provide a path to model.pt checkpoint of your trained model
trained_model_path = 'path/to/your/project/DLProject/experiments/exp_{exp_number}/weights/model.pt'

# You may check your model performance with one of the image from dataset 
# or any other car image on your Google Drive
path_image = 'path/to/local/image'

# Or you also may check it with the any image in the internet
# url_to_image = 'url/to/internet/image'

image = get_input_image_for_inference(local_path=path_to_local_image) 
# image = get_input_image_for_inference(url=url_to_image) 

_, _, scores, masks = model_inference(trained_model_path, image)

# Visualizing results
show_pic_and_pred_instance_masks(image, masks, scores)
```

## Conclusion 

**You've complete your own Deep Learning Project! Congratulations!**

We encourage you to play with the data augmentation, the number of epochs, batch size and many other hyperparameters or add other models to obtain the best performance or just for your interest.

## Submission

To submit your project to the bot you need to compress it to `.zip` with the structure defined in [Project Structure](#project-structure).
              
Upload it to your `Google Drive` and set appropriate rights to submit `DRU-bot` and then you'll receive results.

Nicely done! Waiting for you for our next Data Science courses!
---------------------------------------------------------------------------------------------------------------------------------------------------------
If you have any questions, write `@DRU Team` in Slack!
