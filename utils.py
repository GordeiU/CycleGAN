import logging
import os
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s: [%(levelname)s]: %(message)s")

TMP_STORAGE = os.path.join(".", ".tmp")
TMP_STORAGE_OBESE = os.path.join(TMP_STORAGE, "obese")
TMP_STORAGE_OVERWEIGHT = os.path.join(TMP_STORAGE, "overweight")


########################################################
# Methods for Image DataLoader
########################################################

TRAINED_MODEL_EXT = '.dat'
TRAINED_MODEL_FORMAT = "{}" + TRAINED_MODEL_EXT


##############################################
# Defining all hyperparameters
##############################################

class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

HYPERPARAMETERS = Hyperparameters(
    epoch=0,
    n_epochs=150,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=125,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def convert_to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, normal_data_dir, target_data_dir, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted([os.path.join(normal_data_dir, file_name) for file_name in os.listdir(normal_data_dir)])
        self.files_B = sorted([os.path.join(target_data_dir, file_name) for file_name in os.listdir(target_data_dir)])

        logging.info(f"self.files_A has: {len(self.files_A)} files")
        logging.info(f"self.files_B has: {len(self.files_B)} files")

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        # a % b => a is divided by b, and the remainder of that division is returned.

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        # Finally ruturn a dict
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


########################################################
# Replay Buffer
########################################################

class ReplayBuffer:
    # We keep an image buffer that stores
    # the 50 previously created images.
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # Returns newly added image with a probability of 0.5.
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[
                        i
                    ] = element  # replaces the older image with the newly generated image.
                else:
                    # Otherwise, it sends an older generated image and
                    to_return.append(element)
        return Variable(torch.cat(to_return))


########################################################
# Learning Rate scheduling with `lr_lambda`
########################################################


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        # Below line checks whether the current epoch has exceeded the decay epoch(which is 100)
        # e.g. if current epoch is 80 then max (0, 80 - 100) will be 0.
        # i.e. then entire numerator will be 0 - so 1 - 0 is 1
        # i.e. the original LR remains as it is.
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


########################################################
# Initialize convolution layer weights to N(0,0.02)
########################################################


def initialize_conv_weights_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
