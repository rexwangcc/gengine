"""This libs module contains useful class definitions and helper functions for constructing the BayesianGAN
Architecture. The detailed implementation follows and refers to the below papers and resources:
1. https://arxiv.org/abs/1705.09558
2. https://github.com/andrewgordonwilson/bayesgan
3. https://github.com/vasiloglou/mltrain-nips-2017/blob/master/ben_athiwaratkun/pytorch-bayesgan \
    /Bayesian%20GAN%20in%20PyTorch.ipynb
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import math, os, sys


class NoiseLoss(nn.Module):
    def __init__(self, params, device, scale=None, total=None):
        super().__init__()
        self.device = device
        self.noises = [0.0 * param.data.to(device=device) for param in params]

        if scale: self.scale = scale
        else:
            self.scale = 1.
        self.total = total

    def forward(self, params, scale=None, total=None):
        if not scale: scale = self.scale
        if not total: total = self.total

        assert scale, "scale is not provided, which should follow '2 * SGHMAC_alpha * learning_rate * identity_tensor'"

        noise_loss = 0.0

        for noise, var in zip(self.noises, params):
            _noise = noise.normal_(0, 1)
            # 𝑁(0,2∗𝛼∗𝜂∗𝐼)
            noise_loss = noise_loss + scale * torch.sum(Variable(_noise) * var)
        normed_noise_loss = noise_loss / total
        return normed_noise_loss


class PriorLoss(torch.nn.Module):
    def __init__(self, prior_std=1., total=None):
        super().__init__()
        self.total = total
        self.prior_std = prior_std

    def forward(self, params, total=None):
        if total is None: total = self.total

        prior_loss = 0.0
        for var in params:
            prior_loss = prior_loss + torch.sum((var * var) / (self.prior_std * self.prior_std))
        normed_prior_loss = prior_loss / total
        return normed_prior_loss


class ComplementCrossEntropyLoss(torch.nn.Module):
    """Derived directly from the reference NIPS2017 Workshop."""
    def __init__(self, device, except_index=None, weight=None, ignore_index=-100, size_average=True):
        super().__init__()
        self.except_index = except_index
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.device = device

    def forward(self, input, target=None):
        if target is not None:
            assert not target.requires_grad, "The nn criterions don't compute the gradient w.r.t. targets - please " \
                                             "mark these variables as volatile or not requiring gradients."
        else:
            assert self.except_index is not None, "except_index cannot be None!"
            target = torch.LongTensor(input.data.shape[0]).fill_(self.except_index).to(device=self.device)

        _loss = torch.nn.functional.nll_loss(
                torch.log(1. - torch.nn.functional.softmax(input) + 1e-4),
                target,
                weight=self.weight,
                size_average=self.size_average,
                ignore_index=self.ignore_index)
        return _loss


def weights_init(model):
    """Custom weights initialization called on netG and netD. Refers to the official DCGAN tutorial."""
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)


class PartialDataset(Dataset):
    """Derived directly from the reference NIPS2017 Workshop."""
    def __init__(self, dataset_loader, num_points):
        self.num_points = num_points
        self.dataset_loader = dataset_loader

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.dataset_loader.__getitem__(idx)


class Metrics:
    """Derived and revised from the reference NIPS2017 Workshop."""
    def __init__(self):
        self._value = 0.
        self._mean = 0.
        self._sum = 0.
        self._count = 0.

    def update(self, value, N=1):
        self._value = value
        self._sum += N * value
        self._count += N
        # to be accurate, mean must be computed at the end
        self._mean = self._sum / self._count

    @property
    def value(self):
        return self._value

    @property
    def mean(self):
        return self._mean

    @property
    def sum(self):
        return self._sum

    @property
    def count(self):
        return self._count


def accuracy(output, target, topk=(1,)):
    """Derived directly from the reference NIPS2017 Workshop.

    Computes the precision@k for the specified values of k, used for semi-supervised learning.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_test_accuracy(discriminator, testing_data_loader, device):
    """Derived from the reference NIPS2017 Workshop.

    Make sure that calling `model_d.eval()` before doing the computation!
    """
    top1 = Metrics()
    for i, (input, target) in enumerate(testing_data_loader):
        target = target.to(device=device)
        input = input.to(device=device)
        input_var = torch.autograd.Variable(input.to(device=device), requires_grad=False)
        output = discriminator(input_var)

        probs = output.data[:, 1:]  # discard the zeroth index
        prec1 = accuracy(probs, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))
    print(f'Semi Supervised Learning Test Average Accuracy: {top1.mean:.2f}')
