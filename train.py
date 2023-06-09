import argparse
import datetime
import itertools
import logging
import time
import uuid

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from IPython.display import clear_output
from matplotlib.pyplot import figure
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cyclegan import *
from utils import *

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s: [CycleGAN] [%(levelname)s]: %(message)s")

########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(size, size))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# The reason for doing "np.transpose(npimg, (1, 2, 0))"

# PyTorch modules processing image data expect tensors in the format C × H × W.
# Whereas PILLow and Matplotlib expect image arrays in the format H × W × C
# so to use them with matplotlib you need to reshape it
# to put the channels as the last dimension:

# I could have used permute() method as well like below
# plt.imshow(pytorch_tensor_image.permute(1, 2, 0))

def to_img(x):
    x = x.view(x.size(0) * 2, HYPERPARAMETERS.channels, HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size)
    return x

def plot_output(path, x, y):
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()

##############################################
# SAMPLING IMAGES
##############################################

def save_img_samples(epoch_path, epoch, batches_done):
    """Saves a generated sample from the test set"""
    logging.info(f"batches_done {batches_done}")
    imgs = next(iter(val_dataloader))

    Gen_AB.eval()
    Gen_BA.eval()

    real_A = Variable(imgs["A"].type(Tensor))
    logging.info(real_A.shape)
    fake_B = Gen_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = Gen_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    path = os.path.join(epoch_path, f"{epoch}_{batches_done}.png")

    save_image(image_grid, path, normalize=False)
    return path

##############################################
# Final Training Function
##############################################
def save_model(model, epoch_path):
    torch.save(model.state_dict(), os.path.join(epoch_path, TRAINED_MODEL_FORMAT.format(model.name)))

def train(
    Gen_BA,
    Gen_AB,
    Disc_A,
    Disc_B,
    train_dataloader,
    n_epochs,
    criterion_identity,
    criterion_cycle,
    lambda_cyc,
    criterion_GAN,
    optimizer_G,
    fake_A_buffer,
    fake_B_buffer,
    clear_output,
    optimizer_Disc_A,
    optimizer_Disc_B,
    Tensor,
    sample_interval,
    lambda_id,
):
    # TRAINING
    training_session_path = os.path.join(".", "checkpoints", f"{uuid.uuid4()}")
    os.mkdir(training_session_path)

    prev_time = time.time()
    for epoch in range(HYPERPARAMETERS.epoch, n_epochs):
        epoch_path = os.path.join(training_session_path, f"epoch{epoch + 1}")
        os.mkdir(epoch_path)

        logging.info(f"Epoch {epoch + 1}/{n_epochs}")
        for i, batch in enumerate(train_dataloader):
            logging.info(f"Step: {i + 1}/{len(train_dataloader)}")

            logging.info("Set model input")
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            valid = Variable(
                Tensor(np.ones((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                Tensor(np.zeros((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            #########################
            logging.info("Train Generators")
            #########################

            Gen_AB.train()
            Gen_BA.train()

            """
            PyTorch stores gradients in a mutable data structure. So we need to set it to a clean state before we use it.
            Otherwise, it will have old information from a previous iteration.
            """
            optimizer_G.zero_grad()

            # Identity loss
            # First pass real_A images to the Genearator, that will generate A-domains images
            loss_id_A = criterion_identity(Gen_BA(real_A), real_A)

            # Then pass real_B images to the Genearator, that will generate B-domains images
            loss_id_B = criterion_identity(Gen_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN losses for GAN_AB
            fake_B = Gen_AB(real_A)

            loss_GAN_AB = criterion_GAN(Disc_B(fake_B), valid)

            # GAN losses for GAN_BA
            fake_A = Gen_BA(real_B)

            loss_GAN_BA = criterion_GAN(Disc_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle Consistency losses
            reconstructed_A = Gen_BA(fake_B)

            """
            Forward Cycle Consistency Loss
            Forward cycle loss:  lambda * ||G_BtoA(G_AtoB(A)) - A|| (Equation 2 in the paper)

            Compute the cycle consistency loss by comparing the reconstructed_A images with real real_A  images of domain A.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)

            """
            Backward Cycle Consistency Loss
            Backward cycle loss: lambda * ||G_AtoB(G_BtoA(B)) - B|| (Equation 2 of the Paper)
            Compute the cycle consistency loss by comparing the reconstructed_B images with real real_B images of domain B.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            reconstructed_B = Gen_AB(fake_A)

            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            """
            Finally, Total Generators Loss and Back propagation
            Add up all the Generators loss and cyclic loss (Equation 3 of paper.
            Also Equation I the code representation of the equation) and perform backpropagation with optimization.
            """
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_G.step()

            #########################
            logging.info("Train Discriminator A")
            #########################

            optimizer_Disc_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)

            fake_A_ = fake_A_buffer.push_and_pop(fake_A)

            loss_fake = criterion_GAN(Disc_A(fake_A_.detach()), fake)

            """ Total loss for Disc_A
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_A = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_A.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_A.step()

            #########################
            logging.info("Train Discriminator B")
            #########################

            optimizer_Disc_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_B(real_B), valid)

            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)

            loss_fake = criterion_GAN(Disc_B(fake_B_.detach()), fake)

            """ Total loss for Disc_B
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_B = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_B.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_B.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2

            ##################
            #  Log Progress
            ##################

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            text = "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" % (
                epoch+1,
                n_epochs,
                i+1,
                len(train_dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
                )

            logging.info(text)

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                with open('log.txt', 'a') as log:
                    log.write(f"{text}\n")

                # clear_output()
                save_img_samples(epoch_path=epoch_path, epoch=epoch, batches_done=batches_done)
                # plot_output(save_img_samples(batches_done), 30, 40)

        if epoch % 10 == 0:
            if not os.path.isdir(epoch_path):
                os.mkdir(epoch_path)

            save_model(Gen_AB, epoch_path=epoch_path)
            save_model(Gen_BA, epoch_path=epoch_path)
            save_model(Disc_A, epoch_path=epoch_path)
            save_model(Disc_B, epoch_path=epoch_path)

            logging.info("Models saved")

##############################################
# Execute the Final Training Function
##############################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CycleGAN', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--normal_image_path', default="./data/normal", dest='normal_path', type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--obese', action='store_true')
    group.add_argument('--overweight', action='store_true')

    parser.add_argument('--obese-path', default='./data/obese', type=str, dest='obese_path', help='Path to the obese image file')
    parser.add_argument('--overweight-path', default='./data/overweight', type=str, dest='overweight_path', help='Path to the image dir file')

    parser.add_argument('--model-path', dest='model_path', type=str, required=False)
    parser.add_argument('--epochs', type=int, required=False)

    args = parser.parse_args()

    normal_data_dir = os.path.join(args.normal_path)

    if args.obese:
        target_data_dir = os.path.join(args.obese_path)

    if args.overweight:
        target_data_dir = os.path.join(args.obese_path)

    ##############################################
    logging.info("Dataset paths set")
    ##############################################

    if torch.cuda.is_available():
        cuda = True
        torch.cuda.empty_cache()
    else:
        cuda = False
    logging.info("CUDA is activated" if cuda else "CUDA is not activated")

    """ So generally both torch.Tensor and torch.cuda.Tensor are equivalent. You can do everything you like with them both.
    The key difference is just that torch.Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
    Of course operations on a CPU Tensor are computed with CPU while operations for the GPU / CUDA Tensor are computed on GPU. """
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    ##############################################
    logging.info("Defining Image Transforms to apply")
    ##############################################
    transforms_ = [
        transforms.Resize((HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_dataloader = DataLoader(
        ImageDataset(normal_data_dir, target_data_dir, mode=HYPERPARAMETERS.dataset_train_mode, transforms_=transforms_),
        batch_size=HYPERPARAMETERS.batch_size,
        shuffle=True,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        ImageDataset(normal_data_dir, target_data_dir, mode=HYPERPARAMETERS.dataset_test_mode, transforms_=transforms_),
        batch_size=16,
        shuffle=True,
        num_workers=1,
    )

    ##############################################
    logging.info("SETUP, LOSS, INITIALIZE MODELS and BUFFERS")
    ##############################################

    # Creating criterion object (Loss Function) that will
    # measure the error between the prediction and the target.
    criterion_GAN = torch.nn.MSELoss()

    criterion_cycle = torch.nn.L1Loss()

    criterion_identity = torch.nn.L1Loss()

    input_shape = (HYPERPARAMETERS.channels, HYPERPARAMETERS.img_size, HYPERPARAMETERS.img_size)

    ##############################################
    logging.info("Initializing generator and discriminator")
    ##############################################

    Gen_AB = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks, "Gen_AB")
    Gen_BA = GeneratorResNet(input_shape, HYPERPARAMETERS.num_residual_blocks, "Gen_BA")

    Disc_A = Discriminator(input_shape, "Disc_A")
    Disc_B = Discriminator(input_shape, "Disc_B")

    if cuda:
        Gen_AB = Gen_AB.cuda()
        Gen_BA = Gen_BA.cuda()
        Disc_A = Disc_A.cuda()
        Disc_B = Disc_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    ##############################################
    logging.info("Initializing weights")
    ##############################################

    if args.model_path:
        weights_path = os.path.join(args.model_path)
        d = torch.device("cuda" if cuda else "cpu")
        Gen_AB.load_state_dict(torch.load(os.path.join(weights_path, "Gen_AB.dat"), map_location=d))
        Gen_BA.load_state_dict(torch.load(os.path.join(weights_path, "Gen_BA.dat"), map_location=d))
        Disc_A.load_state_dict(torch.load(os.path.join(weights_path, "Disc_A.dat"), map_location=d))
        Disc_B.load_state_dict(torch.load(os.path.join(weights_path, "Disc_B.dat"), map_location=d))

        logging.info("Initialized weights from the model path")
    else:
        Gen_AB.apply(initialize_conv_weights_normal)
        Gen_BA.apply(initialize_conv_weights_normal)

        Disc_A.apply(initialize_conv_weights_normal)
        Disc_B.apply(initialize_conv_weights_normal)

        logging.info("Initialized random weights")

    ##############################################
    logging.info("Buffers of previously generated samples")
    ##############################################

    fake_A_buffer = ReplayBuffer()

    fake_B_buffer = ReplayBuffer()


    ##############################################
    logging.info("Defining all Optimizers")
    ##############################################
    optimizer_G = torch.optim.Adam(
        itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
        lr=HYPERPARAMETERS.lr,
        betas=(HYPERPARAMETERS.b1, HYPERPARAMETERS.b2),
    )
    optimizer_Disc_A = torch.optim.Adam(Disc_A.parameters(), lr=HYPERPARAMETERS.lr, betas=(HYPERPARAMETERS.b1, HYPERPARAMETERS.b2))

    optimizer_Disc_B = torch.optim.Adam(Disc_B.parameters(), lr=HYPERPARAMETERS.lr, betas=(HYPERPARAMETERS.b1, HYPERPARAMETERS.b2))


    ##############################################
    logging.info("Learning rate update schedulers")
    ##############################################
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(HYPERPARAMETERS.n_epochs, HYPERPARAMETERS.epoch, HYPERPARAMETERS.decay_start_epoch).step
    )

    lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_A,
        lr_lambda=LambdaLR(HYPERPARAMETERS.n_epochs, HYPERPARAMETERS.epoch, HYPERPARAMETERS.decay_start_epoch).step,
    )

    lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_Disc_B,
        lr_lambda=LambdaLR(HYPERPARAMETERS.n_epochs, HYPERPARAMETERS.epoch, HYPERPARAMETERS.decay_start_epoch).step,
    )

    logging.info("Starting training")
    train(
        Gen_BA=Gen_BA,
        Gen_AB=Gen_AB,
        Disc_A=Disc_A,
        Disc_B=Disc_B,
        train_dataloader=train_dataloader,
        n_epochs=args.epochs,
        criterion_identity=criterion_identity,
        criterion_cycle=criterion_cycle,
        lambda_cyc=HYPERPARAMETERS.lambda_cyc,
        criterion_GAN=criterion_GAN,
        optimizer_G=optimizer_G,
        fake_A_buffer=fake_A_buffer,
        fake_B_buffer=fake_B_buffer,
        clear_output=clear_output,
        optimizer_Disc_A=optimizer_Disc_A,
        optimizer_Disc_B=optimizer_Disc_B,
        Tensor=Tensor,
        sample_interval=HYPERPARAMETERS.sample_interval,
        lambda_id=HYPERPARAMETERS.lambda_id,
    )
