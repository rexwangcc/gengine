from __future__ import print_function

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        """The model definition of the generator. This class refers to the PyTorch DCGAN TUTORIAL:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


        Args:
            ngpu (int): The number of GPUs available. If this is 0, code will run in CPU mode. If this number is
                greater than 0 it will run on that number of GPUs.
            nz (int): The size of the latent z vector.
            ngf (int): The depth of feature maps propagated through the discriminator.
            nc (int): The number of the channels of the input images.
        """
        super().__init__()

        self.ngpu = ngpu
        self.main = nn.Sequential(

                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
                nn.Tanh()
        )

    def forward(self, input):
        # Use parallel computing across GPUs if the hardware
        # had multiple available GPUs
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
