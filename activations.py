import torch.nn as nn


def linear(*args, **kwargs):
    return nn.Identity()


def sigmoid(*args, **kwargs):
    return nn.Sigmoid()


def tanh(*args, **kwargs):
    return nn.Tanh()


def relu(inplace=False, *args, **kwargs):
    return nn.ReLU(inplace=inplace)


def gelu(*args, **kwargs):
    return nn.GELU()
