import numpy as np
import itertools
import time
import datetime
import uuid

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
from train import Hyperparameters

from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image
import matplotlib.image as mpimg

from utils import *
from cyclegan import *

import logging


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda = True
        torch.cuda.empty_cache()
    else:
        cuda = False
    logging.info("CUDA is activated" if cuda else "CUDA is not activated")

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    hp = Hyperparameters(
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

    input_shape = (hp.channels, hp.img_size, hp.img_size)

    Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks, "Gen_AB")
    Gen_AB.load_state_dict(torch.load(os.path.join(".", "model", "Gen_AB.dat"), map_location=torch.device('cpu')))
    Gen_AB.eval()
    logging.info("Cycle GAN model loaded")

    transform = transforms.Compose([
        transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(os.path.join(".", "images", "test_image_2.jpg"))
    img = transform(img)
    img = Variable(img.type(Tensor)).to(device='cpu')
    logging.info(img.shape)
    Gen_AB(img)
    # image_tensor = pil_to_model_tensor_transform(pil_loader(os.path.join(args.input, image_name))).to(net.device)
    # net.my_test_single(
    #     image_tensor=image_tensor,
    #     image_name=image_name,
    #     age_group=args.age,
    #     bmi_group=args.bmi_group,
    #     target=results_dest,
    #     watermark=args.watermark
    # )