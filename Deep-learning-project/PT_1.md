# Deep Learning Project &mdash; Part 1: Pipeline
*Hello and welcome to the final project!*

*During this module, you will solve the task of cars images segmentation using the Pytorch Lightning Framework.*

*The aim is to create an effective solution with production-ready code and an understandable structure, which you can reuse for future DL purposes.*

*In this guide, we will go through data processing, building a proper training pipeline with PyTorch Lightning, and performing the inference on raw data using fine-tuned models on a custom dataset. So, let's get started!*

## Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Setup Environment](#setup-environment)
-  [Data Processing](#data-processing)
    -  [Data Overview](#data-overview)
    -  [Defining the Dataset](#defining-the-dataset)
    -  [Data Augmentation](#data-augmentation)
    -  [Defining Pytorch Lightning DataModule](#defining-pytorch-lightning-datamodule)
- [UNet Model](#unet-model)
    -  [Defining UNet model](#defining-unet-model)
    -  [Defining Pytorch Lightning UNet trainer](#defining-pytorch-lightning-unet-trainer)
    -  [Defining Unet Inference](#defining-unet-inference)
-  [Model Training and Inference](#model-training-and-inference)
    -   [Model Training](#model-training)
    -   [Model Inference](#model-inference)
-  [Conclusion](#conclusion)

## Intro

Image segmentation is a task of computer vision that aims at grouping image regions under their respective class labels.

Image segmentation is an extension of image classification where we also need to perform label localization. The model pinpoints where a corresponding object is present by outlining the object's boundary.

Most image segmentation models consist of an encoder-decoder neural network compared to a single encoder network as a classifier.
The encoder encodes a latent space representation of the input, which the decoder decodes to form segment maps by outlining each object’s location in the image.

A typical segment map looks like this:

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/img/segment_map.png">
</div>

## Project Structure

Here is the folder structure. For now, we will only define them, but later we'll consider them in detail:
```
DLProject
        ├── data                                 - contains full data with images and their masks.       
        │    ├── images                          - contains all images.       
        │    │     ├── 0cdf5b5d0ce1_01.jpg
        │    │     ├── 0cdf5b5d0ce1_02.jpg
        │    │     ├── ...
        │    │
        │    └── masks                           - contains all ground truth masks.
        │          ├── 0cdf5b5d0ce1_01.png
        │          ├── 0cdf5b5d0ce1_02.png  
        │          ├── ...
        │
        ├── experiments                          - contains trained models, checkpoints, and logs for every experiment.
        ├── src                                  - contains project source code.
        │   │
        │   ├── configs                          - contains configurable parameters, so you can change them in the future.
        │   │     ├── config.py                  - contains YAML config parser.
        │   │     └── config.yaml                - contains training hyperparameters, folder paths, models architecture, params for optimizers and lr_schedulers.
        │   │                          
        │   ├── models                           - contains all used models.
        │   │     │
        │   │     └── unet                       - contains UNet model folder.
        │   │           ├── unet_model.py        - contains implemented UNet model architecture.
        │   │           ├── unet_trainer.py      - contains Pytorch Lightning Trainer for UNet model.
        │   │           └── unet_inference.py    - contains script to use trained UNet model to make preditions on raw data input.
        │   │
        │   │
        │   ├── processing                       - contains data preprocessing files.
        │   │       ├── dataset.py               - contains scpript to store the data samples and their corresponding target annotation.
        │   │       ├── transforms.py            - contains scpript for data augmentations and transformations.
        │   │       └── datamodule.py            - contains Pytorch Lightning datamodule that encapsulates all the steps needed to process data.
        │   │ 
        │   │                                
        │   └── utils                            - contains utilities functions.
        │         └── utils.py                   - contains visualizing, metrics and commonly used helper functions.
        │
        │
        ├── tests                                - contains pytest tests.
        │    │
        │    ├── models                          - contains segmentation models tests.
        │    │     │
        │    │     └── unet                      - contains UNet model and trainer tests.
        │    │          ├── unet_model_test.py   - contains UNet model tests.
        │    │          └── unet_trainer_test.py - contains UNet trainer tests.
        │    │
        │    └── processing                      - contains data preprocessing tests.
        │           ├── dataset_test.py          - contains the Dataset tests.
        │           ├── transforms_test.py       - contains data transforms and augmentations tests.
        │           └── datamodule_test.py       - contains Datamodule tests.
        │
        ├── requirements.txt                     - contains specifications of required python packages.
        └── train.py                             - contains train model script.
```

**Important Notes**:
1. In order to complete this project, it's better for you to have installed [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows).
2. If you are a student of any university, you can apply for JetBrains Free Educational Licenses and get [PyCharm Professional](https://www.jetbrains.com/community/education/#students) for free.

## Setup Environment 

Here are some important files, that you will use during the work with the project:
1. [tests](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/code/tests.zip) for `./tests`
2. [data](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/code/data.zip) for `./data`
3. [utils](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/code/utils.py.zip) for `./src/utils`
4. [requirements.txt](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/code/requirements.txt.zip) for `./requirements.txt`
5. [configs](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/code/configs.zip) for `./src/configs`

Download them and put into the corresponding directories of your [project](#project-structure). Then install all dependencies: 
```
pip install -r requirements.txt
```

## Data Processing

### Data Overview

The dataset sources from an image masking challenge hosted on [Kaggle from Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge). The objective is to create a model for segmenting high-resolution car images. 

<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/img/dataset_example.gif">
</div>

-   Each image has a resolution of **1918x1280**
-   Each car is presented in **16** different fixed orientations
-   Dataset sample has **1600** images
-   There are only two classes present: 
    -  **class 0**: background
    -  **class 1**: foreground, the car

The dataset for this particular **project contains only the pictures depicting a single car**. In other words, every pixel of every image is binary labeled (car or background). However, in the second part of the project you will implement a model, that may recognize and bound multiple cars on a picture as separate instances **even being trained on single-car images**.

### Defining the Dataset

Let’s write a custom dataset for the Carvana images. After downloading and extracting the `data` file, we have the following folder structure:

```
data/
  images/
    0cdf5b5d0ce1_01.jpg
    0cdf5b5d0ce1_02.jpg
    0cdf5b5d0ce1_03.jpg
    0cdf5b5d0ce1_04.jpg
    ...
  masks/
    0cdf5b5d0ce1_01.png
    0cdf5b5d0ce1_02.png
    0cdf5b5d0ce1_03.png
    0cdf5b5d0ce1_04.png
```

The dataset should inherit from the standard `torch.utils.data.Dataset` class and then you have to implement `__len__` and `__getitem__`.

Our dataset class in the `__getitem__` method should return:
-   **image**: PIL RGB Image of size `(H, W)`
-   **target**: a dict containing the following fields:
    - **image_id** `(Int64Tensor[1])`: an image identifier
    - **mask** `(UInt8Tensor[1, H, W])`: a segmentation mask for corresponding image

Your task is to complete the dataset's class and put it in the `src/processing/dataset.py` file:
```python
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, Any

from src.configs.config import CONFIG


class CarsDataset(Dataset):
    """
    Dataset for training and validation data.
    :param data_root: a path, where data is stored.
    :param transforms: torchvision transforms.
    """

    def __init__(self, data_root: str, transforms: Optional[Any] = None) -> None:
        self.data_root = data_root
        self.transforms = transforms
        self.imgs_list = list(sorted(os.listdir(os.path.join(data_root, CONFIG['dataset']['images_folder'].get()))))
        self.masks_list = list(sorted(os.listdir(os.path.join(data_root, CONFIG['dataset']['masks_folder'].get()))))

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        """
        Retrieve a sample from the dataset.
        :param idx: an index of the sample to retrieve.
        :return: a tuple containing an PIL image and a dict with target annotations data.
        """
        # Get images and masks paths from config
        img_path = os.path.join(self.data_root, CONFIG['dataset']['images_folder'].get(), self.imgs_list[idx])
        mask_path = os.path.join(self.data_root, CONFIG['dataset']['masks_folder'].get(), self.masks_list[idx])

        # Open the image and convert it to "RGB" using img_path
        img = ...
        # Open the mask using mask_path
        mask = ...

        # Convert mask into torch tensor with shape (1, H, W) and dtype torch.uint8
        mask = ...

        image_id = torch.tensor([idx])

        target = {'image_id': image_id, 'mask': mask}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        """
        Retrieve the number of samples in the dataset.
        :return: a len of dataset.
        """
        # Get number of samples in the dataset
        samples_number = ...
        return samples_number

```

After filling the gaps in the dataset class, you should test it with the [`pytest`](https://docs.pytest.org/en/7.0.x/) framework. So, you should specify that we want to use pytest as your tests runner. An example of how this can be done in Pycharm:

`For Windows and UNIX: File --> Settings --> Tools --> Python Integrated Tools --> Default test runner: choose "pytest".`

`For macOS: PyCharm --> Preferences --> Tools --> Python Integrated Tools --> Default test runner: choose "pytest".`

Script with tests is in the `tests/processing/dataset_test.py` file. Highly recommend looking through the tests by yourself and then trying to run test functions one by one.

### Data Augmentation

The main idea is to apply as many relevant data augmentations as possible to extend our dataset. In this project, you will use resize, random crop and horizontal flip transformations to augment your data and satisfy the input size constraints of some models that will be implemented in the future. For example, the model, which will be used in this project, requires the image and mask to have a shape `(512 x 512)`, while all images and masks in our dataset are `(1918 x 1280)`. Therefore, we propose you implement this augmentation approach:

- Convert the PIL image to `torch.Tensor`
- With some probability, apply a random crop:
    - Take a random part of an image and a mask with a shape `(512 x 512)`
    - If the image and mask were not randomly cropped, resize them to `(512 x 512)`
- With some probability, horizontally flip image and mask

Thus, the model will have more data to train and also will be able to segment even a part of a car if it's not fully depicted. After completing your project, you may play with the combinations of transformations and their probabilities and shapes to obtain a better performance.

Let's begin with `ToTensor` class, that convert initial image to `torch.Tensor`:

```python
import torch
import random
from PIL import Image
from torchvision import transforms as T
from typing import Dict, Tuple, Optional, Union

from src.configs.config import CONFIG


class ToTensor:
    """
    Transforms the PIL.Image into torch.Tensor.
    """
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a PIL.Image object.
        :param target: annotation with target image_id and mask.
        :return: a converted torch.Tensor image and its target annotation.
        """
        # Convert PIL Image to pytorch tensor
        image = ...
        return image, target

                . . .
```
Test it with a `test_to_tensor` function in the `tests/processing/transforms_test.py` file.

Implement `Resize` class that will compress the image and the mask to the desired size:

```python
                . . .

class Resize:
    """
    Resizes image and ground truth mask.

    :param resize_to: a size of image and mask to be resized to.
    """
    def __init__(self, resize_to: Tuple[int, int]):
        self.resize_to = resize_to

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]):
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotations.
        :return: a resized image and target annotations with resized mask.
        """
        # Resize image and mask to resize_to shape
        image = ...
        mask = ...
        
        target['mask'] = mask

        return image, target

                . . .
```

Test it with a `test_resize` function in the `tests/processing/transforms_test.py` file.

Implement `RandomCrop` class, that resizes the image and mask to `resize_to` shape and then randomly crop them with shape `crop_to` with some probably `prob` or, if it wasn't done, resize the image to `crop_to` size:

```python
                . . .

class RandomCrop:
    """
    Randomly crop the image and mask with a given probability.
    :param prob: a probability of image and mask being cropped.
    :param crop_to: a size of image and mask to be cropped to.
    """

    def __init__(self, prob: float, crop_to: Tuple[int, int]) -> None:
        self.prob = prob
        self.crop_to = crop_to

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotations.
        :return: cropped image and target annotation with cropped mask.
        """
        if random.random() < self.prob:
            mask = target['mask']

            # Get parameters for applying random crop to crop_to shape
            i, j, h, w = 

            # Randomly crop image and mask
            image = ...
            mask = ...

            target['mask'] = mask
        else:
            # If image and mask were not cropped, resize to crop_to shape
            image, target = ...

        return image, target

                . . .
```

Test it with a `test_random_crop` function in the `tests/processing/transforms_test.py` file.


Implement the `HorizontalFlip` class, that will horizontally flips the image and mask:

```python
                . . .

class HorizontalFlip:
    """
    Horizontally flips the image and mask with a given probability.

    :param prob: a probability of image and mask being flipped.
    """
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: a torch.Tensor image.
        :param target: a dict of target annotation.
        :return: a horizontally flipped image and target annotation with flipped mask.
        """
        if random.random() < self.prob:
            # Flip the image and mask
            image = ...
            mask = ...

            target['mask'] = mask
           
        return image, target
        
                . . .
```
Test it with a `test_horizontal_flip` function in the `tests/processing/transforms_test.py` file.

Paste `Compose` class and `get_transform` function into `src/processing/transforms.py` file. Notice how `get_transform` works: arguments `resize`, `random_crop`, and `hflip` define what exact transformations will be applied, but it's impossible to use both `resize` and `random_crop` simultaneously. This structure allows you to play with augmentations after the project is done or reuse them in your future projects.

```python
                . . .

class Compose:
    """
    Composes several transforms together.
    :param transforms: a list of transforms to compose.
    """
    def __init__(self, transforms: Optional[list] = None) -> None:
        self.transforms = transforms

    def __call__(self, image: Union[Image.Image, torch.Tensor], target: Dict[str, torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param image: an image to be transformed.
        :param target: a dict of target annotation.
        :return: a transformed image and target annotation with transformed mask.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(resize: bool = False, random_crop: bool = False, hflip: bool = False) -> Compose:
    """
    Compose transforms for image amd mask augmentations.
    :param resize: specify, whether to apply image resize or not.
    :param random_crop: specify, whether to apply image random crop or not.
    :param hflip: specify, whether to apply image horizontal flipped or not.
    :return: composed transforms.
    """
    transforms = [ToTensor()]

    if resize:
        transforms.append(Resize(resize_to=(CONFIG['data_augmentation']['resize_to'].get(),
                                            CONFIG['data_augmentation']['resize_to'].get())))
    elif random_crop:
        transforms.append(
            RandomCrop(
                prob=CONFIG['data_augmentation']['random_crop_prob'].get(),
                crop_to=(
                    CONFIG['data_augmentation']['random_crop_crop_to'].get(),
                    CONFIG['data_augmentation']['random_crop_crop_to'].get()
                )
            )
        )
    if hflip:
        transforms.append(HorizontalFlip(prob=CONFIG['data_augmentation']['h_flip_prob'].get()))

    return Compose(transforms)
```

Now we end with the Data Processing part. The next thing we will do is define the full Pytorch Lightning DataModule.

### Defining Pytorch Lightning DataModule

A [DataModule](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html) is the collection of PyTorch `DataLoaders` along with the matching transforms and data processing/loading steps required to prepare the data in a reproducible fashion. So, the `LightningDataModule` makes code reusable across different projects and helps develop dataset agnostic models.

In our datamodule class, we will define several functions: `__init__`, `setup` and `train_dataloader` along with `val_dataloader`.

Let's talk about what they're doing:

1. `__init__`:
    - `data_dir` arg that points to where you store your data
    - `batch_size` arg that points to the number of samples per batch
    - `num_workers` arg points to the number of subprocesses to use for data loading
    - `resize` arg specifies whether to apply image and mask resize or not
    - `random_crop` arg specifies whether to crop the image and mask randomly or not
    - `hflip` arg specifies whether to apply horizontal flip to image and mask or not
2. `setup`:
    - loads in data and prepares PyTorch dataset for train and val split
    - expects a ‘stage’ arg which is used to point to which data split use for ‘fit’
3. `train_dataloader`:
    - returns a PyTorch DataLoader instance that is created by wrapping train data that we prepared in `setup()`
4. `val_dataloader`:
    - returns a PyTorch DataLoader instance that is created by wrapping val data that we prepared in `setup()`

Your task here is to fill the gaps to implement Pytorch Lightning DataModule in `src/processing/datamodule.py`:
```python
import torch
import pytorch_lightning as pl
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader, Subset

from src.processing.dataset import CarsDataset
from src.processing.transforms import get_transform

from src.configs.config import CONFIG


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> \
        Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
    """
    Defines how to collate batches.

    :param batch: a list containing tuples of images and annotations.
    :return: a collated batch. Tuple containing a tuple of images and a tuple of annotations.
    """
    return tuple(zip(*batch))


class CarsDataModule(pl.LightningDataModule):
    """
    LightningDataModule to supply training and validation data.

    :param data_root: a path, where data is stored.
    :param batch_size: a number of samples per batch.
    :param num_workers: a number of subprocesses to use for data loading.
    :param resize: specify, whether to apply image and mask resize or not.
    :param random_crop: specify, whether to randomly crop image and mask or not.
    :param hflip: specify, whether to apply horizontal flip to image and mask or not.
    """
    def __init__(self, data_root: str, batch_size: int,
                 num_workers: int, resize: bool = False,
                 random_crop: bool = False, hflip: bool = False) -> None:
        super().__init__()
        
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.random_crop = random_crop
        self.hflip = hflip
        
        self.full_data = None
        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads in data from file and prepares PyTorch tensor datasets for train and val split.

        :param stage: an argument to separate setup logic for trainer.
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # Initialize CarsDataset with the params given in the __init__ method
            self.full_data = ...

            # The number of samples in the dataset
            n_sample = ...

            # Set split index
            split_idx = round(n_sample * CONFIG['datamodule']['training_set_size'].get())

            # Split full dataset into train and val
            # Hint: Use torch.utils.data.Subset
            self.data_train = ...
            self.data_val = ...

    def train_dataloader(self) -> DataLoader:
        """
        Represents a Python iterable over a train dataset.

        :return: a dataloader for training.
        """
        # Return data train loader object from self.data_train
        # with given batch_size, num_workers, collate_fn
        # and set shuffle as True

        return ...

    def val_dataloader(self) -> DataLoader:
        """
        Represents a Python iterable over a validation dataset.

        :return: a dataloader for validation.
        """
        # Return data val loader object from self.data_val
        # with given batch_size, num_workers, collate_fn
        # and set shuffle as False
        
        return ...
```

To test the `datamodule.py` you should Run the `tests/processing/datamodule_test.py` file.

## UNet Model

After processing our data, we can proceed to the segmentation model. We will use a simple implementation of the UNet model that you've already completed during this course and integrate them into the training logic of the `PyTorchLightning`.

UNet consists of an encoder-decoder scheme: The encoder reduces the spatial dimensions in every layer and increases the channels. On the other hand, the decoder increases the spatial dimensions while reducing the channels. 


<div align="center">
    <img align="center" src="https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/9_deep_learning/deep_learning_project_p1/img/unet_architecture.png">
</div>

In `Lightning`, all modules should be derived from a `LightningModule`, which is a subclass of the `torch.nn.Module`. You may find more details about `LightningModule` in its [documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

For setting up training logic for the model, we will specify `training_step`, `training_epoch_end`, `validation_step`, `validation_epoch_end`, and `configure_optimizers` methods. Luckily, [under the hood](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#under-the-hood), the `Lightning` Trainer will handle the training and validation loop details for you. `Lightning` structures your `PyTorch` code so it can abstract the details of training.

Further, in the second part of the course, you will extend your model zoo with two high-performance models, but the single one will be enough for now)

### Defining UNet model

Let's get started and implement components of UNet architecture in the `src/models/unet/unet_model.py` file:

#### DoubleConv Block

Implement the following:

* `nn.Sequential` should contain `Conv2d(in_channels, out_channels) -> BatchNorm2d -> ReLU -> Conv2d(out_channels, out_channels) -> BatchNorm2d -> ReLU`
* `kernel_size` for `Conv2d` is 3x3
* Use `padding=1`
* Use `inplace=True` for `ReLU`

```python
class DoubleConv(nn.Module):
    """
    Double Convolutional UNet block. Contains two Conv2d layers
    followed by BatchNorm layers and RelU activation functions.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the DoubleConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H, W].
        """
        x = ...
        return x
```

Test the Double Convolutional block in the `tests/models/unet/unet_model_test.py` file running the `test_doubleconv` function.

#### InConv Block

```python
                . . .

class InConv(nn.Module):
    """
    An input block for UNet. A wrapper around DoubleConv block.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        """
        Forward pass through the InConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H, W].
        """
        x = self.conv(x)
        return x

                . . .
```

#### Down Block

Implement in `nn.Sequential` the following:

* Decrease the spatial dimension by half using `MaxPooling`
* Increase the number of channels using `DoubleConv` defined above

```python
                . . .

class Down(nn.Module):
    """
    UNet encoder block, which decreases spatial dimension by two and
    increases the number of channels.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = nn.Sequential(...)

    def forward(self, x):
        """
        Forward pass through the Down block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, out_channels,  H/2, W/2].
        """
        x = ...
        return x

                . . .
```

Test the Down block in the `tests/models/unet/unet_model_test.py` file running the `test_down` function.

#### Up Block

Implement the following:
* Implement `self.up` using `nn.ConvTranspose2d`. Decrease the number of `in_channels` by half (`in_channels` = `out_channels`), use `kernel_size` and `stride` 2x2
* Implement `forward` pass:
    * Upscale `x1` using transpose convolutions
    * Concatenate the inputs across the channel dimension
    * Pass the concatenated tensor through DoubleConv

```python
                . . .

class Up(nn.Module):
    """
    Upscaling UNet block. Takes the output of the previous layer and
    upscales it, increasing the spatial dim by a factor of 2.
    Then, takes the output of the corresponding down layer and concatenates it
    with the previous layer. Finally, passes them through the DoubleConv block.

    :param in_channels: a number of input channels.
    :param out_channels: a number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = ...
        self.conv = ...

    def forward(self, x1, x2):
        """
        Forward pass through the Up block.

        :param x1: an output of the previous upscale layer with shape [batch, out_channels, H, W].
        :param x2: an output of the corresponding down layer with shape [batch, out_channels, 2*H, 2*W].
        :return: an output with shape [batch, out_channels,  2*H, 2*W].
        """
        pass
        return x

                . . .
```

Test the Up block in the `tests/models/unet/unet_model_test.py` file running the `test_up` function.

#### Output Convolution Block

Implement the following:
* Use `Conv2d` with `kernel_size` 1x1 to decrease the number of channels

```python
                . . .

class OutConv(nn.Module):
    """
    UNet output layer, which decreases the number of channels.

    :param in_channels: a number of input channels.
    :param n_classes: a number classes.
    """
    def __init__(self, in_channels, n_classes):
        super(OutConv, self).__init__()
        self.conv = 

    def forward(self, x):
        """
        Forward pass through the OutConv block.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, n_classes,  H, W].
        """
        x = ...
        return x

                . . .
```

Test the Output Convolution block in the `tests/models/unet/unet_model_test.py` file running the `test_outblock` function.

#### Model Class

Implement UNet architecture:
* The progression of channels for encoder blocks:
    * Input channels should be: `in_channels -> 64 -> 128 -> 256 -> 512 -> 512`
    * Output channels should be: `64 -> 128 -> 256 -> 512 -> 512`
* The progression of channels for decoder blocks:
    * Input channels should be: `1024 -> 512 -> 256 -> 128 -> 64`
    * Output channels should be: `256 -> 128 -> 64 -> 64 -> n_classses`

```python
                . . .

class Unet(nn.Module):
    """
    The complete architecture of UNet using layers defined above.

    :param in_channels: a number of input channels.
    :param n_classes: a number classes.
    """
    def __init__(self, in_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = ...
        self.down1 = ...
        self.down2 = ...
        self.down3 = ...
        self.down4 = ...
        self.up1 = ...
        self.up2 = ...
        self.up3 = ...
        self.up4 = ...
        self.out = ...

    def forward(self, x):
        """
        Forward pass through the Neural Network.

        :param x: an input with shape [batch, in_channels,  H, W].
        :return: an output with shape [batch, n_classes,  H, W].
        """
        x1 = ...
        x2 = ...
        x3 = ...
        x4 = ...
        x5 = ...
        x = ...
        x = ...
        x = ...
        x = ...
        out = ...
        return out
```

Test the complete UNet architecture in the `tests/models/unet/unet_model_test.py` file running the `test_full_unet` function.

### Defining Pytorch Lightning UNet trainer

Now let's start implementing the Lightning trainer. You will create the PyTorch Lightning class and integrate the UNet model into the training logic of the `PyTorchLightning` in the `src/models/unet/unet_trainer.py` file..

Firstly, `__init__` method initialize the number of classes the model will predict, an instance of the model, and the criterion to compute the [binary cross-entropy with logits loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html?highlight=bcewithlogitsloss#torch.nn.BCEWithLogitsLoss) between input and target. It is important to note that trainer does not inherit from `nn.Module` as you would commonly do in a pure PyTorch model, but from [`pl.LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html).

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
from typing import Dict, Tuple, Union, List

from src.utils.utils import masks_iou
from src.configs.config import CONFIG
from src.models.unet.unet_model import Unet


class UnetTrainer(pl.LightningModule):
    """
    Pytorch Lightning version of the UNet model.

    :param in_channels: a number of input channels.
    :param n_classes: a number of classes to predict.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.model = Unet(self.in_channels, self.n_classes)
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]],
                      batch_idx: int) -> torch.Tensor:
        pass       
        
    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]],
                        batch_idx: int) -> Dict[str, torch.Tensor]:
        pass    
        
    def training_epoch_end(self, train_losses: List[Dict[str, torch.Tensor]]) -> None:

        pass
        
    def validation_epoch_end(self, val_outputs: List[Dict[str, torch.Tensor]]) -> None:

        pass
        
    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        pass
```

#### Training Step

While in PyTorch, you have to define the entire training loop. The `Lightning` has been automated for you by the `training_step`. However, we still define the steps which will be executed while training.

The `training_step()` method takes as input a single batch of data. It then makes predictions using the UNet model, calculates loss, logs train step loss, and returns it.

Implement the `training_step()` method:

>**Hint**: you need to uncollate the images and targets in the batch. For UNet input, we need a stacked tensor of images and masks. So, we need an `images_tensor` and a `masks_tensor` with shape [batch_dim, C, H, W].
```python
                . . .

    def training_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]],
                  batch_idx: int) -> torch.Tensor:
        """
        Takes a batch and inputs it into the model.
        Retrieves losses after one training step and logs them.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: a dict, which contain the loss after one training step for one batch step.
        """
        images, targets = batch

        # Stack images
        images_tensor = ...
        # Stack masks from targets
        masks_tensor = ...

        # Pass input to the model
        y_hat = ...
        # Calculate loss
        # Hint: convert target input to float()
        loss = ...

        return loss

                . . .

```

Test your `training_step()` method in the `tests/models/unet/trainer.py` file running the `test_training_step` function.

#### Validation Step

This method works exactly like `training_step()` but on validation batch with adding one more validation metric ── masks IoU score.

Implement the `validation_step()` method:
```python
                . . .

    def validation_step(self, batch: Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]],
                    batch_idx: int) -> Tuple[float]:
        """
        Take a batch from the validation dataset and input its images into the model.
        Retrieves losses and masks IoU score after one validation step.

        :param batch: a batch of images and targets with annotations.
        :param batch_idx: an index of the current batch.
        :return: losses and masks IoU score for one batch step.
        """
        images, targets = batch

        # Stack images
        images_tensor = ...
        # Stack masks from targets
        masks_tensor = ...

        # Pass input to the model
        y_hat = self.model(images_tensor)

        # Calculate masks IoU score beetwen target masks and y_hat  
        # Using masks_iou function;
        # num_classes param for masks_iou func is (self.n_classes + 1)
        masks_iou_score = ...
        # Calculate loss
        loss = ...

        # The grid of images and pred masks to log later at the end of the validation epoch
        imgs_grid = get_batch_images_and_pred_masks_in_a_grid(y_hat, images)

        return {'val_loss': loss, 'val_iou': masks_iou_score, 'val_images_and_pred_masks': imgs_grid}

                . . .

```

Test your `validation_step()` method in the `tests/models/unet/trainer.py` file running the `test_validation_step` function.

#### Epochs logging and configure optimizers

`training_epoch_end()` and `validation_epoch_end()` methods here receive as parameters all outputs from `training_step()` and `validation_step` respectively. In `configure_optimizers()` method we just configure [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer and [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) scheduler.

Pass this code into `src/models/unet/trainer.py` file:

```python
                . . .

    def training_epoch_end(self, train_losses: List[Dict[str, torch.Tensor]]) -> None:
        """
        Calculates and logs mean total loss at the end
        of the training epoch with the losses of all training steps.

        :param train_losses: dicts with losses of all training steps.
        :return: mean loss for an epoch.
        """
        loss_epoch = torch.stack([loss['loss'] for loss in train_losses]).mean()

        self.log('train/loss_epoch', loss_epoch.item())

    def validation_epoch_end(self, val_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """
        Calculates and logs mean total loss and average masks IoU at the end of the
        validation epoch with the outputs of all validation steps.

        :param val_outputs: losses and masks IoU scores of all validation steps.
        :return: mean loss and average masks IoU score for an epoch.
        """
        loss_epoch = torch.stack([output['val_loss'] for output in val_outputs]).mean()
        avg_masks_iou = torch.stack([output['val_iou'] for output in val_outputs]).mean()

        self.log('val/loss_epoch', loss_epoch.item(), prog_bar=True)
        self.log('val/val_iou', avg_masks_iou.item(), prog_bar=True)

        # Log predicted masks for validation dataset
        for ind, dict_i in enumerate(val_outputs):
            self.logger.experiment.add_image('Predicted masks on images', dict_i['val_images_and_pred_masks'],
                                             ind)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, object]]:
        """
        Configure the Adam optimizer and StepLR scheduler.

        :return: dict with an optimizer and a lr_scheduler.
        """
        optimizer = Adam(self.model.parameters(),
                         lr=CONFIG['unet']['optimizer']['initial_lr'].get(),
                         weight_decay=CONFIG['unet']['optimizer']['weight_decay'].get())
        lr_scheduler = StepLR(optimizer, step_size=CONFIG['unet']['lr_scheduler']['step_size'].get(),
                              gamma=CONFIG['unet']['lr_scheduler']['gamma'].get())

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
```

The model definition is done, so let's go for training!

### Defining UNet Inference

To run your model with the raw data past the following code block into `src/models/unet/unet_inference.py`:

```python
import torch
import numpy as np
from torchvision import transforms as T

from src.configs.config import CONFIG
from src.models.unet.unet_model import Unet


def model_inference(trained_model_path: str, image: torch.Tensor) -> np.ndarray:
    """
    Loads the entire trained UNet model and predicts segmentation mask for an input image.

    :param trained_model_path: a path to saved model.
    :param image: a torch tensor with a shape [1, 3, H, W].
    :return: predicted mask for an input image.
    """

    model = Unet(CONFIG['unet']['model']['in_channels'].get(),
                 CONFIG['unet']['model']['n_classes'].get())
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()

    resize_to = (
        CONFIG['data_augmentation']['resize_to'].get(),
        CONFIG['data_augmentation']['resize_to'].get()
    )

    resized_image = T.Resize(resize_to)(image)

    with torch.no_grad():
        prediction = model(resized_image)
        mask = torch.sigmoid(prediction)

    orig_size = (image.shape[2], image.shape[3])
    resized_mask_to_original_image_size = T.Resize(orig_size)(mask)

    return resized_mask_to_original_image_size.numpy()

```

## Model Training and Inference

### Model Training

Here is the function for training the model, put it into `./train.py`:

```python
import os
import torch
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config import CONFIG
from src.processing.datamodule import CarsDataModule

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


def train_model(exp_number: int, model: Any, batch_size: int = 4,
                max_epochs: int = 1, use_resize: bool = False,
                use_random_crop: bool = False, use_hflip: bool = False) -> None:
    """
    Trains a Segmentation model based on a given config.

    :param exp_number: number of training experiment.
    :param model: an instance of the model we are going to train
    :param batch_size: a number of samples per batch.
    :param max_epochs: a number of epochs to train model.
    :param use_resize: specify, whether to apply image resize or not.
    :param use_random_crop: specify, whether to apply image resize and random crop or not.
    :param use_hflip: specify, whether to apply image horizontal flip or not.
    """
    datamodule = CarsDataModule(
        data_root=CONFIG['dataset']['data_root'].get(),
        batch_size=batch_size,
        num_workers=CONFIG['dataloader']['num_workers'].get(),
        resize=use_resize,
        random_crop=use_random_crop,
        hflip=use_hflip
    )

    experiment_folder = 'exp_' + str(exp_number)

    # Creates experiment folders to save there logs and weights
    # Weights folder:
    weights_folder_path = os.path.join(os.path.join(CONFIG['trainer']['logs_and_weights_root'].get(),
                                                    experiment_folder), CONFIG['trainer']['weights_folder'].get())
    os.makedirs(weights_folder_path, exist_ok=True)
    # Logs folder:
    logs_folder_path = os.path.join(os.path.join(CONFIG['trainer']['logs_and_weights_root'].get(),
                                                 experiment_folder), CONFIG['trainer']['logger_folder'].get())
    os.makedirs(logs_folder_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_folder_path,
        every_n_epochs=1,
        monitor='val/loss_epoch',
        auto_insert_metric_name=False,
        filename='sample-cars-segm-model-epoch{epoch:02d}-val_loss_epoch{val/loss_epoch:.3f}')

    trainer = pl.Trainer(max_epochs=max_epochs,
                         accelerator=accelerator,
                         devices=1,
                         logger=TensorBoardLogger(
                             save_dir=os.path.join(
                                 CONFIG['trainer']['logs_and_weights_root'].get(),
                                 experiment_folder
                             ),
                             name=CONFIG['trainer']['logger_folder'].get()),
                         log_every_n_steps=CONFIG['trainer']['log_every_n_steps'].get(),
                         callbacks=[checkpoint_callback]
                         )
    trainer.fit(model, datamodule)

    torch.save(model.model.state_dict(), os.path.join(weights_folder_path, 'model.pt'))
```

In order to train the model quickly, we need GPU. If you do not have any GPUs on your own you may use ones from Google Colab this way:

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

Now it's time to train your UNet model:

```python
from train import train_model
from src.models.unet.unet_trainer import UnetTrainer

model = UnetTrainer()
train_model(exp_number=1, model=model, batch_size=4, max_epochs=5, use_resize=False, use_random_crop=True, use_hflip=True)
```

Then check your logs in TensorBoard:
```python
%reload_ext tensorboard
%tensorboard --logdir /path/to/your/project/DLProject/experiments
```

### Model Inference

During training each model, you will save a model checkpoint for every epoch and a fully trained model after the entire train in the `experiments/exp_{exp_number}/weights` folder. So let's check the model predictions on a raw data using the `model_inference` function from `src/models/unet/unet_inference.py`

```python
from src.utils.utils import get_input_image_for_inference
from src.models.unet.unet_inference import model_inference
from src.utils.utils import show_pic_and_semantic_mask

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

## Submission

To submit your project to the bot you need to compress it to `.zip` with the structure defined in [Project Structure](#project-structure).
              
Upload it to your `Google Drive` and set appropriate rights to submit `DRU-bot` and then you'll receive results.

## Conclusion 

Congrats! You've done with the end-to-end image segmentation pipeline!
We encourage you to play with the data augmentation, number of epochs, batch size and many other hyperparameters to obtain the best performance or just for your interest.

**What's next to do?**

In the second part of the project you will extend the abilities of your segmentation implementing two more models and integrating them into this pipeline.

Good luck and have a fun!
---------------------------------------------------------------------------------------------------------------------------------------------------------
And, as always, if you have any questions, feel free to write `@DRU Team` in Slack!
