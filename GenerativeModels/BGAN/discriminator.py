from __future__ import print_function

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, nc=3, num_class=1):
        """The model definition of the discriminator. This class refers to the PyTorch DCGAN TUTORIAL:
        https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html


        Args:
            ngpu (int): The number of GPUs available. If this is 0, code will run in CPU mode. If this number is
                greater than 0 it will run on that number of GPUs.
            ndf (int): The depth of feature maps propagated through the discriminator.
            nc (int): The number of the channels of the input images.
            num_class (int): The number of the classification categories(classes), or logits. By default it's 1.
        """
        super().__init__()

        self.ngpu = ngpu
        self.num_class = num_class
        self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, num_class, 2, 1, 0, bias=False),
        )
        if self.num_class == 1: self.main.add_module('prob', nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(input.size(0), self.num_class).squeeze(1)
