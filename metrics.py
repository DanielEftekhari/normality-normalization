import torch

import utils


def discrete_entropy(probs):
    return -torch.sum(utils.where(probs > 0, probs * torch.log(probs)), dim=-1)


def max_discrete_entropy(c):
    return torch.log(c)


def max_continuous_entropy(sigma):
    return 0.5 * torch.log(2 * torch.pi * torch.square(sigma)) + 0.5


def accuracy(cm):
    return (torch.trace(cm) / torch.sum(cm)).item()


def confusion_matrix(y_hat, y, c_dim):
    cm = torch.zeros((c_dim, c_dim), dtype=torch.int32, device=y_hat.device)
    cm.index_put_((y, y_hat), torch.tensor([1], dtype=torch.int32, device=y_hat.device), accumulate=True)
    return cm


def accuracy_topk(y_hat, y, topk=(1,)):
    maxk = max(topk)
    batch_size = y.size(0)

    _, pred = y_hat.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1. / batch_size))
    return res
