from sklearn.utils import shuffle
from torch.utils.data import Subset


class TargetTransform:
    def __init__(self, map_class):
        self.map_class = map_class

    def __call__(self, target):
        return self.map_class[target]


class Subset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = [dataset.targets[i] for i in indices]


def split_data(x, y, train, train_fraction=0.85, random_state=2):
    n = x.shape[0]
    x, y = shuffle(x, y, random_state=random_state)
    if train:
        return x[0: int(train_fraction * n)], y[0: int(train_fraction * n)]
    else:
        return x[int(train_fraction * n):], y[int(train_fraction * n):]
