import argparse
import logging
import os
from os.path import join as path_join

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image

from cyclegan import *
from utils import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s: [%(levelname)s]: %(message)s")

if __name__ == "__main__":
    logging.info("CycleGAN generation started")

    parser = argparse.ArgumentParser(description='CycleGAN', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--obese', action='store_true')
    parser.add_argument('--overweight', action='store_true')

    args = parser.parse_args()

    input_shape = (HYPERPARAMETERS.channels, HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size)

    if args.obese:
        logging.info("Loading normal to obese generator...")
        Gen_normal_to_obese = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        Gen_normal_to_obese.load_state_dict(torch.load(path_join(".", "models", "Gen_normal_to_obese.dat"), map_location=torch.device('cpu')))
        Gen_normal_to_obese.eval()
        logging.info("Loaded normal to obese generator")

    if args.overweight:
        logging.info("Loading normal to overweight generator...")
        # Gen_normal_to_overweight = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        # Gen_normal_to_overweight.load_state_dict(torch.load(path_join(".", "models", "Gen_normal_to_overweight.dat"), map_location=torch.device('cpu')))
        # Gen_normal_to_overweight.eval()
        logging.info("Loaded normal to overweight generator")

    Tensor = torch.Tensor

    transform = transforms.Compose([
        transforms.Resize((HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(path_join(".", "images", "test_image_0.jpg"))
    img = transform(img)
    img = Variable(img.type(Tensor)).to(device='cpu')
    img = img.unsqueeze(0)
    logging.info("Image prepared for CycleGAN")

    if not os.path.exists(TMP_STORAGE):
        create_path(TMP_STORAGE)
        create_path(TMP_STORAGE_OBESE)
        create_path(TMP_STORAGE_OVERWEIGHT)

        logging.info("Created a .tmp dir for results")

    if args.obese:
        obese = Gen_normal_to_obese(img)
        save_image(obese, path_join(TMP_STORAGE_OBESE, 'test.jpg'), normalize=True)
        logging.info("Generated obese image")

    if args.overweight:
        # overweight = Gen_normal_to_overweight(img)
        # save_image(overweight, path_join(TMP_STORAGE_OBESE, 'test.jpg'), normalize=True)
        logging.info("Generated overweight image")
