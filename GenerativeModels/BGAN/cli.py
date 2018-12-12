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
from libs import weights_init, ComplementCrossEntropyLoss, PriorLoss, NoiseLoss, PartialDataset, Metrics, accuracy, compute_test_accuracy
import numpy as np


def semi_main(options):
    print('\nSemi-Supervised Learning!\n')

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

    # 6. Prepare the datasets and split it for semi-supervised learning
    if options.dataset != 'cifar10':
        raise NotImplementedError('Semi-supervised learning only support CIFAR10 dataset at the moment!')
    test_data_loader, semi_data_loader, train_data_loader = _prepare_semi_dataset(options)

    # 7. Set the parameters
    ngpu = int(options.ngpu)  # num of GPUs
    nz = int(options.nz)  # size of latent vector, also the number of the generators
    ngf = int(options.ngf)  # depth of feature maps through G
    ndf = int(options.ndf)  # depth of feature maps through D
    nc = int(options.nc)  # num of channels of the input images, 3 indicates color images
    M = int(options.mcmc)  # num of SGHMC chains run concurrently
    nd = int(options.nd)  # num of discriminators
    nsetz = int(options.nsetz)  # num of noise batches

    # 8. Special preparations for Bayesian GAN for Generators

    # In order to inject the SGHMAC into the training process, instead of pause the gradient descent at
    # each training step, which can be easily defined with static computation graph(Tensorflow), in PyTorch,
    # we have to move the Generator Sampling to the very beginning of the whole training process, and use
    # a trick that initializing all of the generators explicitly for later usages.
    Generator_chains = []
    for _ in range(nsetz):
        for __ in range(M):
            netG = Generator(ngpu, nz, ngf, nc).to(device)
            netG.apply(weights_init)
            Generator_chains.append(netG)

    logging.info(f'Showing the first generator of the Generator chain: \n {Generator_chains[0]}\n')

    # 9. Special preparations for Bayesian GAN for Discriminators
    assert options.dataset == 'cifar10', 'Semi-supervised learning only support CIFAR10 dataset at the moment!'

    num_class = 10 + 1

    # To simplify the implementation we only consider the situation of 1 discriminator
    # if nd <= 1:
    #     netD = Discriminator(ngpu, ndf, nc, num_class=num_class).to(device)
    #     netD.apply(weights_init)
    # else:
    # Discriminator_chains = []
    # for _ in range(nd):
    #     for __ in range(M):
    #         netD = Discriminator(ngpu, ndf, nc, num_class=num_class).to(device)
    #         netD.apply(weights_init)
    #         Discriminator_chains.append(netD)

    netD = Discriminator(ngpu, ndf, nc, num_class=num_class).to(device)
    netD.apply(weights_init)
    logging.info(f'Showing the Discriminator model: \n {netD}\n')

    # 10. Loss function
    criterion = nn.CrossEntropyLoss()
    all_criterion = ComplementCrossEntropyLoss(except_index=0, device=device)

    # 11. Set up optimizers
    optimizerG_chains = [
        optim.Adam(netG.parameters(), lr=options.lr, betas=(options.beta1, 0.999)) for netG in Generator_chains
    ]

    # optimizerD_chains = [
    #     optim.Adam(netD.parameters(), lr=options.lr, betas=(options.beta1, 0.999)) for netD in Discriminator_chains
    # ]
    optimizerD = optim.Adam(netD.parameters(), lr=options.lr, betas=(options.beta1, 0.999))
    import math
    # 12. Set up the losses for priors and noises
    gprior = PriorLoss(prior_std=1., total=500.)
    gnoise = NoiseLoss(params=Generator_chains[0].parameters(), device=device,
                       scale=math.sqrt(2 * options.alpha / options.lr), total=500.)
    dprior = PriorLoss(prior_std=1., total=50000.)
    dnoise = NoiseLoss(params=netD.parameters(), device=device,
                       scale=math.sqrt(2 * options.alpha * options.lr), total=50000.)

    gprior.to(device=device)
    gnoise.to(device=device)
    dprior.to(device=device)
    dnoise.to(device=device)

    # In order to let G condition on a specific noise, we attach the noise to a fixed Tensor
    fixed_noise = torch.FloatTensor(options.batchSize, options.nz, 1, 1).normal_(0, 1).to(device=device)
    inputT = torch.FloatTensor(options.batchSize, 3, options.imageSize, options.imageSize).to(device=device)
    noiseT = torch.FloatTensor(options.batchSize, options.nz, 1, 1).to(device=device)
    labelT = torch.FloatTensor(options.batchSize).to(device=device)
    real_label = 1
    fake_label = 0

    # 13. Transfer all the tensors and modules to GPU if applicable
    # for netD in Discriminator_chains:
    #     netD.to(device=device)
    netD.to(device=device)

    for netG in Generator_chains:
        netG.to(device=device)
    criterion.to(device=device)
    all_criterion.to(device=device)

    # ========================
    # === Training Process ===
    # ========================

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    stats = []
    iters = 0

    try:
        print("\nStarting Training Loop...\n")
        for epoch in range(options.niter):
            top1 = Metrics()
            for i, data in enumerate(train_data_loader, 0):
                # ##################
                # Train with real
                # ##################
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                # label = torch.full((batch_size,), real_label, device=device)

                inputT.resize_as_(real_cpu).copy_(real_cpu)
                labelT.resize_(batch_size).fill_(real_label)

                inputv = torch.autograd.Variable(inputT)
                labelv = torch.autograd.Variable(labelT)

                output = netD(inputv)
                errD_real = all_criterion(output)
                errD_real.backward()
                D_x = 1 - torch.nn.functional.softmax(output).data[:, 0].mean().item()

                # ##################
                # Train with fake
                # ##################
                fake_images = []
                for i_z in range(nsetz):
                    noiseT.resize_(batch_size, nz, 1, 1).normal_(0, 1)  # prior, sample from N(0, 1) distribution
                    noisev = torch.autograd.Variable(noiseT)
                    for m in range(M):
                        idx = i_z * M + m
                        netG = Generator_chains[idx]
                        _fake = netG(noisev)
                        fake_images.append(_fake)
                # output = torch.stack(fake_images)
                fake = torch.cat(fake_images)
                output = netD(fake.detach())

                labelv = torch.autograd.Variable(torch.LongTensor(fake.data.shape[0]).to(device=device).fill_(fake_label))
                errD_fake = criterion(output, labelv)
                errD_fake.backward()
                D_G_z1 = 1 - torch.nn.functional.softmax(output).data[:, 0].mean().item()

                # ##################
                # Semi-supervised learning
                # ##################
                for ii, (input_sup, target_sup) in enumerate(semi_data_loader):
                    input_sup, target_sup = input_sup.to(device=device), target_sup.to(device=device)
                    break
                input_sup_v = input_sup.to(device=device)
                target_sup_v = (target_sup + 1).to(device=device)
                output_sup = netD(input_sup_v)
                err_sup = criterion(output_sup, target_sup_v)
                err_sup.backward()
                pred1 = accuracy(output_sup.data, target_sup + 1, topk=(1,))[0]
                top1.update(value=pred1.item(), N=input_sup.size(0))

                errD_prior = dprior(netD.parameters())
                errD_prior.backward()
                errD_noise = dnoise(netD.parameters())
                errD_noise.backward()
                errD = errD_real + errD_fake + err_sup + errD_prior + errD_noise
                optimizerD.step()

                # ##################
                # Sample and construct generator(s)
                # ##################
                for netG in Generator_chains:
                    netG.zero_grad()
                labelv = torch.autograd.Variable(torch.FloatTensor(fake.data.shape[0]).to(device=device).fill_(real_label))
                output = netD(fake)
                errG = all_criterion(output)

                for netG in Generator_chains:
                    errG = errG + gprior(netG.parameters())
                    errG = errG + gnoise(netG.parameters())
                errG.backward()
                D_G_z2 = 1 - torch.nn.functional.softmax(output).data[:, 0].mean().item()
                for optimizerG in optimizerG_chains:
                    optimizerG.step()

                # ##################
                # Evaluate testing accuracy
                # ##################
                # Pause and compute the test accuracy after every 10 times of the notefreq
                if iters % 10 * int(options.notefreq) == 0:
                    # get test accuracy on train and test
                    netD.eval()
                    compute_test_accuracy(discriminator=netD,
                                          testing_data_loader=test_data_loader,
                                          device=device)
                    netD.train()

                # ##################
                # Note down
                # ##################
                # Report status for the current iteration
                training_status = f"[{epoch}/{options.niter}][{i}/{len(train_data_loader)}] Loss_D: {errD.item():.4f} " \
                                  f"Loss_G: " \
                                  f"{errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}" \
                                  f" | Acc {top1.value:.1f} / {top1.mean:.1f}"
                print(training_status)

                # Save samples to disk
                if i % int(options.notefreq) == 0:
                    vutils.save_image(real_cpu,
                                      f"{options.outf}/real_samples_epoch_{epoch:{0}{3}}_{i}.png",
                                      normalize=True)
                    for _iz in range(nsetz):
                        for _m in range(M):
                            gidx = _iz * M + _m
                            netG = Generator_chains[gidx]
                            fake = netG(fixed_noise)
                            vutils.save_image(fake.detach(),
                                              f"{options.outf}/fake_samples_epoch_{epoch:{0}{3}}_{i}_z{_iz}_m{_m}.png",
                                              normalize=True)

                    # Save Losses statistics for post-mortem
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())
                    stats.append(training_status)

                    # # Check how the generator is doing by saving G's output on fixed_noise
                    # if (iters % 500 == 0) or ((epoch == options.niter - 1) and (i == len(data_loader) - 1)):
                    #     with torch.no_grad():
                    #         fake = netG(fixed_noise).detach().cpu()
                    #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    iters += 1
            # TODO: find an elegant way to support saving checkpoints in Bayesian GAN context
    except Exception as e:
        print(e)

        # save training stats no matter what kind of errors occur in the processes
        _save_stats(statistic=G_losses, save_name='G_losses', options=options)
        _save_stats(statistic=D_losses, save_name='D_losses', options=options)
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
        if not (options.disable_cuda or options.ngpu == 0):
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


def _prepare_semi_dataset(options):
    if options.dataset == 'cifar10':
        train = dset.CIFAR10(root=options.dataroot,
                               download=True,
                               transform=transforms.Compose([
                                   # transforms.Resize(options.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

        test = dset.CIFAR10(root=options.dataroot,
                             download=True,
                             train=False,
                             transform=transforms.Compose([
                                 # transforms.Resize(options.imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

        # pinning memory to speed up transferring data to GPU:
        # https: // pytorch.org / docs / master / notes / cuda.html  # use-pinned-memory-buffers
        test_loader = torch.utils.data.DataLoader(test,
                                                  batch_size=options.batchSize,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=int(options.workers))
        semi_loader = torch.utils.data.DataLoader(PartialDataset(train, options.semiSample),
                                                  batch_size=options.batchSize,
                                                  shuffle=True,
                                                  num_workers=int(options.workers))
        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=options.batchSize,
                                                   shuffle=True,
                                                   num_workers=int(options.workers))
        return test_loader, semi_loader, train_loader

def _save_stats(statistic, save_name, options):
    with open(f"{str(options.outf).strip('/')}/{str(save_name).strip('_')}_{datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.txt", "w") as f:
        for item in statistic:
            f.write(f"{item}\n")


if __name__ == '__main__':
    """Usage: `python cli.py --dataset='celeba' --dataroot='path_to_gengine/GenerativeModels/dataset/celeba' --niter=50 --outf="./training" --niter=1`
    
    and 
    
    python cli.py --dataset='cifar10' --semi --dataroot='./dataset/' --outf="./training" --niter=1
    """

    # Establish the CLI options parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Only support cifar10 and lsun at the moment.')
    parser.add_argument('--dataroot', required=True, help='Path to dataset root folder.')
    parser.add_argument('--semi', action='store_true', help='If specified, will perform semi-supervised learning. Only support training with CIFAR-10 dataset at the moment.')
    parser.add_argument('--semiSample', type=int, default=1000, help='Number of samples will be used in Semi-Supervised learning. [default 1000]')
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers [default 2].')
    parser.add_argument('--batchSize', type=int, default=64, help='The input batch size. [default 64]')
    parser.add_argument('--imageSize', type=int, default=64, help='The height / width of the input image to network. [default 64]')
    parser.add_argument('--nsetz', type=int, default=1, help='The number of the {z} sets for marginalization.')
    parser.add_argument('--nz', type=int, default=10, help='The size of the latent z vector. [default 10]')
    parser.add_argument('--nd', type=int, default=1, help='In the context of BGAN, indicates the number of discriminators. [default 1]')
    parser.add_argument('--alpha', type=float, default=0.001, help='The friction term for SGHMC.  [default 0.001]')
    parser.add_argument('--mcmc', type=int, default=10, help='The number of the SGHMC chains for sampling. [default 10]')
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
    parser.add_argument('--outf', default='.', help='Folder to output images; checkpoints are not supported in BGAN at the moment!')
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

    if cli_options.semi:
        # Start the training process and perform semi-supervised learning
        semi_main(cli_options)
    else:
        raise NotImplementedError('We only have semi-supervised learning implementation at the moment.')
