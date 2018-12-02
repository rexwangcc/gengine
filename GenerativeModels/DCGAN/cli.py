from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import logging
from generator import Generator
from discriminator import Discriminator
import datetime


def main(options):
    # 1. Make sure the options are valid argparse CLI options indeed
    assert isinstance(options, argparse.Namespace)

    # 2. Set up the logger
    logging.basicConfig(level=str(options.loglevel).upper())

    # 3. Make sure the output dir `outf` exists
    _check_out_dir(options)

    # 4. Set the random state
    _set_random_state(options)

    # 5. Configure CUDA and Cudnn, set the global `device` for PyTorch
    device = _configure_cuda(options)

    # 6. Prepare the datasets
    data_loader = _prepare_dataset(options)

    # 7. Set the parameters
    ngpu = int(options.ngpu)  # num of GPUs
    nz = int(options.nz)  # size of latent vector
    ngf = int(options.ngf)  # depth of feature maps through G
    ndf = int(options.ndf)  # depth of feature maps through D
    nc = int(options.nc)  # num of channels of the input images, 3 indicates color images

    # 8. Initialize (or load checkpoints for) the Generator model
    netG = Generator(ngpu, nz, ngf, nc).to(device)
    netG.apply(weights_init)
    if options.netG != '':
        logging.info(f'Found checkpoint of Generator at {options.netG}, loading from the saved model.\n')
        netG.load_state_dict(torch.load(options.netG))
    logging.info(f'Showing the Generator model: \n {netG}\n')

    # 9. Initialize (or load checkpoints for) the Discriminator model
    netD = Discriminator(ngpu, ndf, nc).to(device)
    netD.apply(weights_init)
    if options.netD != '':
        logging.info(f'Found checkpoint of Discriminator at {options.netG}, loading from the saved model.\n')
        netD.load_state_dict(torch.load(options.netD))
    logging.info(f'Showing the Discriminator model: \n {netD}\n')

    # ========================
    # === Training Process ===
    # ========================

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    stats = []
    iters = 0

    # Set the loss function to Binary Cross Entropy between the target and the output
    # See https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(options.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=options.lr, betas=(options.beta1, 0.999))

    print("\nStarting Training Loop...\n")
    for epoch in range(options.niter):
        for i, data in enumerate(data_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            training_status = f"[{epoch}/{options.niter}][{i}/{len(data_loader)}] Loss_D: {errD.item():.4f} Loss_G: " \
                              f"{errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {(D_G_z1/D_G_z2):.4f}"
            print(training_status)

            if i % int(options.notefreq) == 0:
                vutils.save_image(real_cpu,
                                  f"{options.outf}/real_samples.png",
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  f"{options.outf}/fake_samples_epoch_{epoch:{0}{3}}.png",
                                  normalize=True)

                # Save Losses statistics for post-mortem
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                stats.append(training_status)

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == options.niter - 1) and (i == len(data_loader) - 1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1


        # do checkpointing
        torch.save(netG.state_dict(), f"{options.outf}/netG_epoch_{epoch}.pth")
        torch.save(netG.state_dict(), f"{options.outf}/netD_epoch_{epoch}.pth")

    # save training stats
    _save_stats(statistic=G_losses, save_name='G_losses', options=options)
    _save_stats(statistic=D_losses, save_name='G_losses', options=options)
    _save_stats(statistic=stats, save_name='Training_stats', options=options)


def _check_out_dir(options):
    """Make sure the output dir `outf` exists."""
    try:
        os.makedirs(options.outf)
    except OSError:
        pass


def _set_random_state(options):
    """Set the random state."""
    if options.manualSeed is None:
        options.manualSeed = random.randint(1, 10000)
    logging.info(f"Using Random Seed: {options.manualSeed}")
    random.seed(options.manualSeed)
    torch.manual_seed(options.manualSeed)


def _configure_cuda(options):
    """Configure CUDA and Cudnn."""
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        if not (options.disable_cuda or options.ngpu==0):
            logging.info("Using GPU with CUDA.\n")
            device = torch.device('cuda:0')
            logging.info(f"GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            logging.warning("You have available CUDA device(s), so you should probably run without --disable-cuda!\n")
            device = torch.device('cpu')
            logging.info("Using CPU.\n")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU.\n")
    return device


def _prepare_dataset(options):
    """Prepare the datasets."""
    if options.dataset in ['imagenet', 'folder', 'lfw', 'celeba']:
        # folder dataset
        dataset = dset.ImageFolder(root=options.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(options.imageSize),
                                       transforms.CenterCrop(options.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif options.dataset == 'mnist':
        raise NotImplementedError
    elif options.dataset == 'lsun':
        dataset = dset.LSUN(root=options.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(options.imageSize),
                                transforms.CenterCrop(options.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif options.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=options.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(options.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    elif options.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, options.imageSize, options.imageSize),
                                transform=transforms.ToTensor())
    assert dataset

    # Create the data_loader instance from the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batchSize,
                                             shuffle=True, num_workers=int(options.workers))
    return dataloader


def weights_init(model):
    """Custom weights initialization called on netG and netD."""
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


def _save_stats(statistic, save_name, options):
    with open(f"{str(options.outf).strip('/')}/{str(save_name).strip('_')}_{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}", "w") as f:
        for item in statistic:
            f.write(f"{item}\n")


if __name__ == '__main__':
    """Usage: `python cli.py --dataset='celeba' --dataroot='/Users/chengche/Mint/CodePlayground/MyCodes/Github Projects/gengine/GenerativeModels/dataset/celeba' --niter=50 --outf="./training" --niter=1`"""

    # Establish the CLI options parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | celeba | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='Path to dataset root folder.')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers [default 2].')
    parser.add_argument('--batchSize', type=int, default=64, help='The input batch size. [default 64]')
    parser.add_argument('--imageSize', type=int, default=64, help='The height / width of the input image to network. [default 64]')
    parser.add_argument('--nz', type=int, default=100, help='The size of the latent z vector. [default 100]')
    parser.add_argument('--nc', type=int, default=3, help='The number of the channels of the input images. [default 3]')
    parser.add_argument('--ngf', type=int, default=64, help='The depth of feature maps carried through the generator. [default 64]')
    parser.add_argument('--ndf', type=int, default=64, help='The depth of feature maps propagated through the discriminator. [default 64]')
    parser.add_argument('--niter', type=int, default=25, help='The number of epochs to train for. [default 25]')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate, [default 0.0002].')
    parser.add_argument('--beta1', type=float, default=0.5, help='The beta1 for Adam optimizers. [default 0.5]')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA. You do not want this flag if you have GPU!')
    parser.add_argument('--ngpu', type=int, default=1, help='The number of GPUs available. If this is 0, code will run '
                                                            'in CPU mode. If this number is greater than 0 it will run on '
                                                            'that number of GPUs. [default 1]')
    parser.add_argument('--netG', default='', help="Path to netG (to continue training from model checkpoint)")
    parser.add_argument('--netD', default='', help="Path to netD (to continue training from model checkpoint)")
    parser.add_argument('--outf', default='.', help='Folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='Set manual seed with PyTorch for generating random numbers.')
    parser.add_argument('--loglevel', default='info', help='debug | info | warning | error | critical; [default info]')
    parser.add_argument('--notefreq', type=int, default=50, help='The frequency of noting down the traning status, every X iterations. [default 50]')
    cli_options = parser.parse_args()


    # Create a timestamp-based folder for outputs
    cli_options.outf = f"{str(cli_options.outf).strip('/')}/{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"

    # List out all of the parsed CLI options
    prefix = "=>"
    print("Using the following parameters:")
    for arg in vars(cli_options):
        print(f"{prefix}\t{arg}={getattr(cli_options, arg)}")

    # Start the training process
    main(cli_options)
