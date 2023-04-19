import argparse
import logging
import os
import shutil
from os.path import join as path_join

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image

from cyclegan import *
from utils import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s: [CycleGAN] [%(levelname)s]: %(message)s")

def combine_images(input_dirs=[TMP_STORAGE_NORMAL, TMP_STORAGE_OVERWEIGHT, TMP_STORAGE_OBESE], output_path=TMP_STORAGE+"result.jpg"):
    images = []

    for dir in input_dirs:
        for file in os.listdir(dir)[::-1]:
            if file.endswith(('.jpg')):
                img = Image.open(os.path.join(dir, file))
                images.append(img)

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    total_width = max_width * IMAGES_PER_ROW
    total_height = max_height * ((len(images) + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW)

    new_image = Image.new('RGB', (total_width, total_height))
    
    x_offset, y_offset = 0, 0
    for idx, img in enumerate(images):
        new_image.paste(img, (x_offset, y_offset))
        x_offset += max_width
        
        if (idx + 1) % IMAGES_PER_ROW == 0:
            x_offset = 0
            y_offset += max_height

    new_image.save(output_path)

    logging.info("Output images combined")

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
    parser.add_argument('--combine', action='store_true')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to the image file')
    group.add_argument('--image-dir-path', type=str, dest='image_dir', help='Path to the image dir file')

    parser.add_argument('--output-dir', type=str)

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
    cycle_gan_path = os.path.dirname(os.path.abspath(__file__))

    if args.obese:
        Gen_normal_to_obese = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        Gen_normal_to_obese.load_state_dict(torch.load(path_join(cycle_gan_path, "models", "Gen_normal_to_obese.dat"), map_location=torch.device('cpu')))
        Gen_normal_to_obese.eval()
        logging.info("Loaded normal to obese generator")

    if args.overweight:
        Gen_normal_to_overweight = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks)
        Gen_normal_to_overweight.load_state_dict(torch.load(path_join(cycle_gan_path, "models", "Gen_normal_to_overweight.dat"), map_location=torch.device('cpu')))
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
    create_path(TMP_STORAGE_NORMAL)

    logging.info("Created a .tmp dir for results")

    if args.image:
        generate_images(path_join(args.image), args.obese, args.overweight, Gen_normal_to_obese, Gen_normal_to_overweight)

    if args.image_dir:
        for file_name in os.listdir(os.path.join(args.image_dir)):
            shutil.copyfile(path_join(args.image_dir, file_name), path_join(TMP_STORAGE_NORMAL, file_name))
            generate_images(path_join(args.image_dir, file_name), args.obese, args.overweight, Gen_normal_to_obese, Gen_normal_to_overweight)

        if args.combine:
            if args.output_dir:
                combine_images(output_path=path_join(args.output_dir, "result.jpg"))
            else:
                combine_images()

    if args.clean:
        os.system('rm -rf ' + TMP_STORAGE)
        logging.info("Cleaned the tmp directory")