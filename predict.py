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

def generate_images(image_path,
                    obese: bool,
                    overweight: bool,
                    Gen_normal_to_obese: GeneratorResNet,
                    Gen_normal_to_overweight: GeneratorResNet):
    file_name = os.path.basename(image_path)
    img = Image.open(image_path)
    img = transform(img)
    img = Variable(img.type(Tensor)).to(device='cpu')
    img = img.unsqueeze(0)
    logging.info("Image prepared for CycleGAN")

    if obese:
        obese_image = Gen_normal_to_obese(img)
        save_image(obese_image, path_join(TMP_STORAGE_OBESE, file_name), normalize=True)
        logging.info("Generated obese image")

    if overweight:
        overweight_image = Gen_normal_to_overweight(img)
        save_image(overweight_image, path_join(TMP_STORAGE_OVERWEIGHT, file_name), normalize=True)
        logging.info("Generated overweight image")

if __name__ == "__main__":
    logging.info("CycleGAN generation started")

    parser = argparse.ArgumentParser(description='CycleGAN', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--obese', action='store_true')
    parser.add_argument('--overweight', action='store_true')
    parser.add_argument('--clean', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to the image file')
    group.add_argument('--image-dir-path', type=str, dest='image_dir', help='Path to the image dir file')

    args = parser.parse_args()

    if args.image:
        if not os.path.isfile(args.image):
            raise ValueError(f"{args.image} is not a file")

        extension = args.image.split('.')[-1]
        if extension != "jpg":
            raise TypeError(f"Unsupported type {extension} the only supported type is jpg")

    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            raise NameError(f"Given path does not exist: {args.image_dir}")

    input_shape = (HYPERPARAMETERS.channels, HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size)

    #TODO: Potentially update the paths to models to be args in the arg parse
    if args.obese:
        logging.info("Loading normal to obese generator...")
        Gen_normal_to_obese = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        Gen_normal_to_obese.load_state_dict(torch.load(path_join(".", "models", "Gen_normal_to_obese.dat"), map_location=torch.device('cpu')))
        Gen_normal_to_obese.eval()
        logging.info("Loaded normal to obese generator")

    if args.overweight:
        logging.info("Loading normal to overweight generator...")
        Gen_normal_to_overweight = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        Gen_normal_to_overweight.load_state_dict(torch.load(path_join(".", "models", "Gen_normal_to_overweight.dat"), map_location=torch.device('cpu')))
        Gen_normal_to_overweight.eval()
        logging.info("Loaded normal to overweight generator")

    Tensor = torch.Tensor

    transform = transforms.Compose([
        transforms.Resize((HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if os.path.exists(TMP_STORAGE):
        os.system('rm -rf ' + TMP_STORAGE)

    create_path(TMP_STORAGE)
    create_path(TMP_STORAGE_OBESE)
    create_path(TMP_STORAGE_OVERWEIGHT)

    logging.info("Created a .tmp dir for results")

    if args.image:
        generate_images(path_join(args.image), args.obese, args.overweight, Gen_normal_to_obese, Gen_normal_to_overweight)

    if args.image_dir:
        for file_name in os.listdir(os.path.join(args.image_dir)):
            generate_images(path_join(args.image_dir, file_name), args.obese, args.overweight, Gen_normal_to_obese, Gen_normal_to_overweight)

    if args.clean:
        os.system('rm -rf ' + TMP_STORAGE)
        logging.info("Cleaned the tmp directory")