# Face Editing with StyleGAN2-ada

*Greetings, and welcome to StyleGAN project! Here you will learn how to use a pretrained StyelGAN checkpoint
to generate and edit realistic human faces. We will also implement a cool GUI as a bonus!
We tried to make it as fun and practical as possible, so we hope you will like it :)*

![interface4](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/interface4.png)

# Table Of Contents
-  [Intro](#intro)
-  [Project Structure](#project-structure)
-  [Before we start](#before-we-start)
-  [Main Components](#main-components)
    -  [StyleGAN](#stylegan)
    -  [Generator](#generator)
    -  [Settings](#settings)
    -  [Shifter](#shifter)
    -  [Align Images](#align-images)
    -  [Projector](#projector)
    -  [Controllers](#controllers)
    -  [GUI](#gui)
- [Run](#run)
  

## Intro

With the recent advancments in **GAN**s (**G**enerative **A**dverserial **N**etworks), it 
finally became possible to generate realistic human faces. In this project, we will learn how
to generate faces and edit facial attributes using pretrained **[StyleGAN2-ada](https://arxiv.org/abs/1812.04948)** checkpoint
that was trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset. We will then use
**[PySimpleGUI](https://pysimplegui.readthedocs.io/en/latest/)** graphical library to make
a visual interface for our app. 

> 📘 **Materials**
> 
> Before you start this project read the following materials:
> * Original StyleGAN [paper](https://arxiv.org/abs/1812.04948)
> * An introduction [guide](https://realpython.com/pysimplegui-python/) to PySimpleGUI.

## Project Structure

Here is the overall structure of our app. Don't worry we will walk you through every component 
of this app.
```
face_edit_app
    ├──  core                   - this folder controls the generation and projection 
    │   ├── align_faces.py      - aligns the face using landmark detection model
    │   ├── generator.py        - class for generating with StyleGAN
    │   ├── projector.py        - defines functions to get the latent vector for a given image
    |   └── shifter.py          - class for changing the direction of latent vectors
    │
    │
    ├──  generated              - this is where we will save our generated data
    │   ├── ptoj.mp4            - the video of projection
    │   ├── proj.png            - projected image
    |   └── projected_w.npy     - corresponding latent vector
    │
    │
    ├── gui                     - this folder contains all components of user interface.
    │   ├── layouts             - defines windows, buttons, sliders, etc.
    │   │   ├── interface.py    - layout for the main panel
    │   │   ├── sliders.py      - layout of sliders (part of interface
    │   │   └── project.py      - layout for projection window
    │   │   
    │   └── main.py             - overall interface logic
    │   
    │   
    ├── input_imgs              - put your own images here
    │   ├── myimg.png           - example of an image
    │   └── myimg_aligned.png   - aligned image created during projection pipeline
    │ 
    │ 
    ├── models                  - put model checkpoints here
    │   ├── ffhq.pkl            - StyleGAN checkpoint
    │   └── face_landmarks.dat  - Landmark detection model checkpoint
    │ 
    │ 
    ├── settings                - here you can store different constant values, model parameters, etc.
    │   ├── __ini__.py          - initializing our settings as a package
    │   └── config.py           - configuration of generation, projection, user interface, etc.
    │ 
    │ 
    ├── stylegan2-ada-pytorch   - original StyleGAN2 repository
    │   └── ...
    │ 
    │ 
    ├── utils                   - utility functions are stored here
    │   ├── __init__.py         - initialize utils as a package
    │   └── helpers.py          - helper functions
    │ 
    │ 
    ├── vectors                 - put vectors for editing here
    │   ├── age.npy             - latent vector for changing age
    │   ├── gender.npy          - latent vector for changing gender
    │   └── ...
    │ 
    │ 
    ├── controller.py           - high-level functionality for the app
    |
    ├── run.py    				- application run file.
    |
    └── requirements.txt		- list of libraries for the project
```
Now it's time for you to define the structure by carrying it to your local machine!

## Before we start

First of all, we have to install the needed packages. You will need
`cmake` for image aligning. You can use `homebrew` if you have MacOS, `apt-get` if you use Linux,
or `Chocolatey` for Windows to install this package.

We will be working from the `face_edit_app` directory:
```
cd face_edit_app
```

Then let's create `requierments.txt`:
```
numpy
torch==1.7.1
pillow
pysimplegui
click
requests
pyspng
ninja
imageio-ffmpeg==0.4.3
dlib
``` 

Most of these packages should be already installed, but if you don't have some of them, run:
```
pip install -r requirements.txt
```

After you are done installing needed packages, clone original StyleGAN2-ada repository
and download the checkpoint

```
mkdir models
cd models
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
cd ..
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

### StyleGAN

Before we jump into the implementation details, let's briefly discuss the basics of StyleGAN. 
This part is just a recap of a few important points from the paper, so read the original paper
before starting this project. 

![stylegan](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/StyleGAN.png)

#### StyleGAN Latent Space

In the **regular** GAN, changing the input latent vector will inevitably lead to big random changes in the output. 
Thus, the latent space of a regular GAN is **entangled**, meaning that it is hard to find 
correlations between latent vectors and some distinct features of generated images like age, gender, etc. 

However, **StyleGAN** has two latent spaces, Z and W, to resolve this issue. The vectors from the input
space, Z, go through a mapping network into a new **disentangled** space W, in which vectors have much 
more meaningful features. For example, we can manipulate vectors from space W to make a face look
older or more masculine. 

#### Generation

In **StyleGAN**, generation starts from a constant input that "merges" with style vector w through
a mechanism called AdaIn at each layer of the Generator. The Generator layers themselves are simple 
Convolutional layers followed by Upsampling layers. The randomness in the generated image comes from 
additional noise input that is being added to the input at each step of the generation. 

#### Latent Space Exploration

Since the latent space w in StyleGAN is disentangled, things that are similar in appearance should be "close" 
to each other in the latent space. For example, if you have a vector w1 that generates a face of a young woman, 
and w2 that generates a face of the old man, you can take w_avg = (w1 + w2)/2 and get an image of a middle-aged person
that is visually similar to the two previously generated people. Moreover, we can transition/interpolate between points 
in this space to create smooth animations.

#### Truncation Trick

To improve the quality of the generated images, StyleGAN uses a truncation trick. The idea is to shift every vector w
into the direction of the average vector w:

w_shifted = psi * (w - w_avg) + w_avg

where psi is a constant in range [0, 1] that controls the amount of shifting.

In a sense, the truncation trick is a trade-off between the quality and the diversity of the generated images. The smaller is psi, 
the closer is w vectors to the average w vector, which means that generated images will have a good quality, but they 
will look very similar. On the contrary, if psi is close to one, we will often generate w vectors that are far
from the average w vector, which means that there is a higher chance of getting a poor quality image, but images will
have more diversity.

## Main Components

**Important Notes:**

1. In order to complete this project, it's better for you to have installed [**PyCharm**](https://www.jetbrains.com/pycharm/download/)
2. If you are a student of any university you can apply for [**JetBrains Free Educational Licenses**](https://www.jetbrains.com/community/education/#students) and get **PyCharm Professional** for free (only for the period of study)
3. Also we recommend using [**Anaconda**](https://www.anaconda.com/) package manager

### Settings

Let's create a file where we will keep all our configurations `settings/config.py`:

```python
import torch


class Generation:
    default_psi = 0.75
    model_path = "models/ffhq.pkl"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Projection:
    generation_dir = "generated/"
    alignment_dir = "input_imgs/"
    alignment_model = 'models/shape_predictor_68_face_landmarks.dat'
    num_steps = 100
    save_video = True
    seed = 305


class Shifting:
    vectors_path = "vectors/"
    extension = ".npy"


class GUIConfig:
    theme_name = "DarkTeal9"
    display_size = (400, 400)
    shift_range = (-10, 10)
    vector_names = (
        "age",
        "eye_distance",
        "eye_eyebrow_distance",
        "eye_ratio",
        "eyes_open",
        "gender",
        "lip_ratio",
        "mouth_open",
        "mouth_ratio",
        "nose_mouth_distance",
        "nose_ratio",
        "nose_tip",
        "pitch",
        "roll",
        "smile",
        "yaw",
    )


class Config:
    generation = Generation
    shifting = Shifting
    gui = GUIConfig
    projection = Projection
```

As you can see our `Config` class is actually a composition of 4 separate configs. This makes it
easier to manage different configurations for different aspects of the app.

Also, don't forget to create `settings/__init__.py`:

```python
from .config import *
```

### Utils

Next, we will define utility functions that we will be using later. We define the first two functions for you, but you have two define the last two on your own.

**Hints:**
* use `x = (x + 1)*127.5` to convert the image range
* don't forget to change data type to `uint8` for an image

File `utils/helpers.py`:

```python
import io
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_img_bits(image):
    """
    Converts PIL Image it to bits
    """
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


def plot_image(image, title=None):
    """
    Displays an image using matplotlib
    """
    plt.figure(figsize=(8,8))
    if title is not None:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def load_numpy(path, device):
    """
    Loads numpy array and converts it to torch tensor
    """
    # load from path
    data = 
    # convert to torch tensor and pss to device
    data = 
    return data


def convert_image(image, size):
    """
    Converts generated image to a displayable format
    @param image: a generated image of shape [N, H, W] in range [-1, 1]
    @return: postprocessed image in shape [H, W, N] in range [0, 255] (dtype=uint8)
    """
    # change the order of channels
    image = 
    # renormalize
    image = 
    # convert to NumPy
    image = 
    # convert to PIL
    image = 
    # resize to a desired size
    image = 
    return image
```

Also create `utils/__init__.py`:

```
from .helpers import *
```

### Generation

Next, let's make a wrapper around the StyleGAN model to simplify the inference. Your job
will be to define `truncate_w` and `get_w` methods of the `Generator`

Make a file `core/generator.py`:

```python
import torch
import pickle
import numpy as np
from settings import Config

import sys
sys.path.append('stylegan2-ada-pytorch')


class Generator:
    def __init__(self, pickle_path=Config.generation.model_path, device=Config.generation.device):
        self.device = device
        with open(pickle_path, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(device)
            self.G.eval()

    def truncate_w(self, w, truncation_psi=1):
        """
        Performs linear interpolation between a given w and the average w vector
        truncation_psi=1 means no truncation
        """
        w_avg = self.G.mapping.w_avg
        # perform truncation
        w = 
        return w

    def get_z(self, seed):
        """
        Generates latent vector z from a random seed
        """
        z = np.random.RandomState(seed).randn(1, self.G.z_dim)
        return z

    def get_w(self, z, truncation_psi=1):
        """
        Generates w vector using latent vector z
        """
        z = torch.tensor(z).to(self.device)
        with torch.no_grad():
            # get w using G.mapping(z, None)
            w = 
            # perform truncation
            w = 
        return w

    def get_img(self, w):
        """
        Generates image using latent vector w
        """
        with torch.no_grad():
            img = self.G.synthesis(w, noise_mode='const', force_fp32=True)[0]
        return img
```

Let's test the generator. You can use Jupyter lab, terminal, or any other python environment 
to perform testing.

```python
from core.generator import Generator
from utils import convert_image, plot_image

model = Generator()

z = model.get_z(123)
w = model.get_w(z, truncation_psi=0.5)
img = model.get_img(w)
img = convert_image(img, (500, 500))
plot_image(img)
```

The result should look something like that:

![geneated1](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/generated1.png)

### Shifter 

There are ways to learn latent directions (both supervised, and unsupervised) in the latent space to control features. 
People have already open-sourced some directional latent vectors for StyleGAN2 that allow us to "move" in the latent 
space and control a particular feature.

* Supervised Method of learning these latent directions:
> "We first collect multiple samples (image + latent) from our model and manually classify the 
> images for our target attribute (e.g. smiling VS not smiling), trying to guarantee proper class representation balance. 
> We then train a model to classify or regress on our latents and manual labels. At this point we can use the learned 
> functions of these support models as transition directions" - 
> [5agado's Blog](https://towardsdatascience.com/stylegan-v2-notes-on-training-and-latent-space-exploration-e51cf96584b3)

* Unsupervised Method: 
[Unsupervised Discovery of Interpretable Directions in the GAN Latent Space](https://arxiv.org/abs/2002.03754)

We will be using learned vectors from [Robert Luxemburg](https://twitter.com/robertluxemburg/status/1207087801344372736)
Download the vectors from [here](https://hostb.org/NCM) and put the content of the zip archive
into `vectors` directory.


Using these vectors we can  move in a latent direction using the following operation: 

w = w + latent_direction * magnitude

* **w** is our latent code from W space
* **latent_direction** is a learnt directional vector that is of shape (18, 512). 
This vector tells you where to move in the latent space to control a feature, but not how much to move.
* **magnitude** is the amount to move in the direction of latent_direction

Let's define `Shifter` class that will perform the shifting of the vectors. 
`core/shifter.py`:
```python
import os
from settings import Config
from utils import load_numpy


class Shifter:
    def __init__(self, vectors_dir=Config.shifting.vectors_path, ext=Config.shifting.extension):
        self.fnames = [file for file in os.listdir(vectors_dir) if file.endswith(ext)]
        self.vectors = {}
        for file in self.fnames:
            path = os.path.join(vectors_dir, file)
            name = file.replace(ext, '')
            # laod numpy vectors and pass to device
            vec = 
            # unsqueeze to add "batch" dimension
            vec = 
            self.vectors[name] = vec

    def __call__(self, w, direction, amount):
        """
        Shifts latent vector w in the given direction
        @param w: input vector
        @param direction: name of a key in vectors dictionary
        @param amount: scale factor for direction
        @return: shifted vector
        """
        # perform shifting
        w = 
        return w
```

Let's test `Shifter` class:

```python
from core.generator import Generator
from core.shifter import Shifter
from utils import convert_image, plot_image

model = Generator()
shifter = Shifter()

z = model.get_z(123)
w = model.get_w(z, truncation_psi=0.5)
w = shifter(w, 'age', 5)
img = model.get_img(w)
img = convert_image(img, (500, 500))
plot_image(img)
```

The result should look something like that:
![geneated1](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/generated2.png)

### Align Images

Next, we will try to do something different. Instead of simply generating an image using some random vector, we will 
generate an input vector using the real image. Even though it may seem easy it's not a straightforward process, and 
the first step would be to make our image as close as possible to images in FFHQ dataset. To do so, we will be using
a landmark detection model to align our image based on key points on the face. 

First, run the following code to download the model.
``` 
cd models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dv shape_predictor_68_face_landmarks.dat.bz2
cd ..
```


Then copy the following code to `core/align_faces.py` (you don't have to implement anything here, but we encourage you 
to go through this piece of code on your own):
```python
"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html
requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import numpy as np
import PIL
import PIL.Image
import sys
import os
import glob
import scipy
import scipy.ndimage
import dlib

from settings import Config

# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor(Config.projection.alignment_model)


def get_landmark(filepath):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm


def align_face(filepath, outdir=Config.projection.alignment_dir):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 1024
    transform_size = 4096
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    filename = filepath.split('/')[-1]
    name, ext = filename.split('.')
    path = os.path.join(outdir, f'{name}_aligned.{ext}')
    img.save(path)
    return path
```

### Projection

In short, to get the projection, we will run optimization loop to optimize for the vector w that generates an image
closest to the input image.

For as many iterations we will:

1. Ask the generator to generate some output from a starting latent vector.
2. Take the generator's output image, and take your target image, put them both through a VGG16 model (image feature extractor).
3. Take the generator's output image features from the VGG16.
4. Take the target image features from the VGG16.
5. Compute the loss on the difference of features!
6. Backpropagate to optimize the input vector

Create a file `core/projector.py` and copy the following code (you don't have to implement anything here, but we encourage you 
to go through this piece of code on your own):

```python
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""
import sys
sys.path.append('stylegan2-ada-pytorch')
import dnnlib
import legacy


import copy
import os
from time import perf_counter

import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from settings import Config

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, force_fp32=True, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

def run_projection(
    target_fname,
    G,
    outdir         = Config.projection.generation_dir,
    save_video     = Config.projection.save_video,
    seed           = Config.projection.seed,
    num_steps      = Config.projection.num_steps,
    device         = Config.generation.device
):
    """
    Project given image to the latent space of pretrained network pickle.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), force_fp32=True, noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    projected_w = projected_w_steps[-1].unsqueeze(0)
    img = G.synthesis(projected_w, force_fp32=True, noise_mode='const')[0]
    synth_image = (img + 1) * (255/2)
    synth_image = synth_image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.save(f'{outdir}/projected_w.npy', projected_w.cpu().numpy())
    return img, projected_w
```

### Controllers

Finally, let's put all of this functioanlity together in one `Controller` class.

Create `controllers.py` and finish the following code:

```python
from core.generator import Generator
from core.shifter import Shifter
from core.projector import run_projection
from core.align_faces import align_face

from utils import convert_image, get_img_bits, load_numpy
from PIL import Image
from settings import Config


class Controller:
    def __init__(self):
        self.generator = Generator()
        self.shifter = Shifter()
        self.w = None

    def generate_img_from_seed(self, seed, psi):
        """
        Generates bits of an image using seeded z vector and truncation psi
        """
        # get z from seed
        z = 
        # get w from z
        self.w = 
        # get img from w
        img = 
        # convert image (use size=Config.gui.display_size)
        img = 
        # convert image to bits
        img_bits = 
        return img_bits

    def generate_img_from_z_vec(self, path, psi):
        """
        Generates bits of an image using a path to a z vector and truncation psi
        (also saves w vector)
        """
        # load z from path 
        z = 
        # get w from z
        self.w = 
        # get img from w
        img = 
        # convert image (use size=Config.gui.display_size)
        img = 
        # convert image to bits
        img_bits = 
        return img_bits

    def generate_img_from_w_vec(self, path):
        """
        Generates bits of an image using a path to a w vector and truncation psi
        (also saves w vector)
        """
        # load w from path
        self.w = 
        # get img from w
        img = 
        # convert image (use size=Config.gui.display_size)
        img = 
        # convert image to bits
        img_bits = 
        return img_bits

    def trasnform_img(self, directions, psi):
        """
        Transforms saved w vector and generates new transformed image bits.
        @param directions: dictionary where key is direction and value is shifting amount (i.e. 'age':5.0)
        @param psi: truncation psi value.
        @return: transformed image bits
        """
        # perform truncation
        w_prime = 
        for direction, amount in directions.items():
            # shift
            w_prime = 
        # generate the image
        img = 
        # convert image (use size=Config.gui.display_size)
        img = 
        # convert image to bits
        img_bits = 
        return img_bits

    def read_path(self, path):
        """
        Read an image from a path converts it to display size and converts it to bits
        """
        img = Image.open(path)
        img = img.resize(Config.gui.display_size)
        img_bits = get_img_bits(img)
        return img_bits

    def align(self, path):
        """
        Aligns the image using provided path and returns bits of aligned image and its path
        """
        # align the image
        aligned_path = 
        # read image bit from the path
        aligned_bits = 
        return aligned_bits, aligned_path

    def project(self, path):
        """
        Runs projection and returns bits of the generated image (also saves w vector)
        """
        # run projrction
        img, self.w = 
        # convert image (size = Config.gui.display_size)
        img = 
        # convert image to bits
        projected_bits = 
        return projected_bits

```

### GUI

For Graphical User Interface (GUI), we will be using a package called PySimpleGUI. The workflow is the following:

1. Create a static layout
2. Define a Window
3. Run an infinite loop 

Let's define static layouts first

`gui/layout/sliders.py`:

```python
import PySimpleGUI as sg
from settings import Config
sg.theme(Config.gui.theme_name)

sliders = [[sg.Slider(Config.gui.shift_range, orientation="h", resolution=.01,
                      default_value=0.0, size=(30, 15), key=f"{name}"),
            sg.Text(name, auto_size_text=True)] for name in Config.gui.vector_names]
sliders += [[sg.Button("Transform", key="TRANSFORM"),
             sg.Button("Reset", key="RESET")]]
```

`gui/layouts/interface.py`:

```python
import PySimpleGUI as sg
from settings import Config
from .sliders import sliders
sg.theme(Config.gui.theme_name)

right_column = [[sg.Text("Select the generation method:")],
                [sg.Button("seed", key="SEED"),
                 sg.Button("z vector", key="Z_VEC"),
                 sg.Button("w vector", key="W_VEC"),
                 sg.Button("project", key="PROJECT")],
                [sg.Slider((0, 1), orientation="h", resolution=.01,
                           default_value=1, size=(30, 15), key="PSI"), sg.Text("truncation psi")],
                [sg.Column(sliders, visible=False, key="SLIDERS")]]

left_column = [
    [sg.Text("Original generated image", key="ORIGINAL_CAPTION", visible=False)],
    [sg.Image(key="ORIGINAL_IMAGE", size=Config.gui.display_size, visible=False)],
    [sg.Text("Modified image", key="MODIFIED_CAPTION", visible=False)],
    [sg.Image(key="MODIFIED_IMAGE", size=Config.gui.display_size, visible=False)]
]

main_layout = [[sg.Column(left_column),
                sg.Column(right_column)]]
```

`gui/layouts/project.py`:

```python
import PySimpleGUI as sg
from settings import Config

sg.theme(Config.gui.theme_name)

projection_layout = [
    [sg.Text("Choose a file for projection"),
     sg.In(size=(40, 1), enable_events=True, key="FILE"),
     sg.FileBrowse(),
     sg.Submit()],
    [sg.Column([[sg.Image(key="ORIGINAL", size=Config.gui.display_size)],
                [sg.Text("Original Image")]]),
     sg.Column([[sg.Image(key="ALIGNED", size=Config.gui.display_size)],
                [sg.Text("Aligned Image")]]),
     sg.Column([[sg.Image(key="PROJECTED", size=Config.gui.display_size)],
                [sg.Text("Projected Image")]])]
]
```

Now, let's define the logic that will run our interface.

`gui/main.py`:

```python
import PySimpleGUI as sg
from settings import Config
from controller import Controller
from .layouts.interface import main_layout
from .layouts.project import projection_layout

sg.theme(Config.gui.theme_name)


def projection_window(controller):
    project_window = sg.Window("Running projection", projection_layout)
    while True:
        event, values = project_window.read()
        print(event, values)
        if event in (sg.WIN_CLOSED, "EXIT"):
            break
        elif event == "Submit":
            original_img_bits = controller.read_path(values["FILE"])
            project_window["ORIGINAL"].update(data=original_img_bits)
            project_window.refresh()
            aligned_img_bits, aligned_path = controller.align(values["FILE"])
            project_window["ALIGNED"].update(data=aligned_img_bits)
            project_window.refresh()
            projected_img_bits = controller.project(aligned_path)
            project_window["PROJECTED"].update(data=projected_img_bits)
            project_window.refresh()
            sg.popup_ok("Projection has finished", keep_on_top=True)
            break
    project_window.close()
    return projected_img_bits


def main():
    window_main = sg.Window("Generator", main_layout)
    controller = Controller()

    while True:
        event, values = window_main.read()

        print(event, values)
        if event in (sg.WIN_CLOSED, "EXIT"):
            break

        elif event in ("SEED", "Z_VEC", "W_VEC", "PROJECT"):
            if event == "SEED":
                seed = int(sg.popup_get_text("Enter z seed: ", title="input"))
                generated_img_bits = controller.generate_img_from_seed(seed, values["PSI"])
            elif event == "Z_VEC":
                path = sg.popup_get_file("Enter the path to z vector", title="input")
                generated_img_bits = controller.generate_img_from_z_vec(path, values["PSI"])
            elif event == "W_VEC":
                path = sg.popup_get_file("Enter the path to w vector", title="input")
                generated_img_bits = controller.generate_img_from_w_vec(path)
            elif event == "PROJECT":
                generated_img_bits = projection_window(controller)

            # update visibility
            window_main["ORIGINAL_CAPTION"].update(visible=True)
            window_main["ORIGINAL_IMAGE"].update(visible=True)
            window_main["SLIDERS"].update(visible=True)
            # update data
            window_main["ORIGINAL_IMAGE"].update(data=generated_img_bits)

        elif event == "TRANSFORM":
            directions = {key: values[key] for key in Config.gui.vector_names}
            bits = controller.trasnform_img(directions, values["PSI"])

            # update visibility
            window_main["MODIFIED_CAPTION"].update(visible=True)
            window_main["MODIFIED_IMAGE"].update(visible=True)
            # update data
            window_main["MODIFIED_IMAGE"].update(data=bits)

        elif event == "RESET":
            for name in Config.gui.vector_names:
                window_main[name].Update(value=0)
            window_main["MODIFIED_CAPTION"].update(visible=False)
            window_main["MODIFIED_IMAGE"].update(visible=False)

    window_main.close()

```

## Run

Yay! We succesfully finished writing all components of this project, let's just define a runner file to run the project.

`run.py`:

```python
from gui.main import main

if __name__ == "__main__":
    main()
```

The interface of the project should look like this:

![interface1](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/interface1.png)
![interface2](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/interface2.png)
![interface3](https://dru.fra1.digitaloceanspaces.com/DL_pytorch/static/ntbk_images/interface3.png)
