import os
from PIL import Image
import torchvision.datasets as tv_datasets


class MNIST(tv_datasets.MNIST):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)


class CIFAR10(tv_datasets.CIFAR10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        super(CIFAR10, self).__init__(root, train, transform, target_transform, download)


class CIFAR100(tv_datasets.CIFAR100):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        super(CIFAR100, self).__init__(root, train, transform, target_transform, download)


class SVHN(tv_datasets.SVHN):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        if train: split = 'train'
        else: split = 'test'
        super(SVHN, self).__init__(root, split, transform, target_transform, download)
        self.targets = self.labels


class ImageNet(tv_datasets.ImageNet):
    def __init__(self,
                 root,
                 train=True,
                 *args, **kwargs):
        if train: split = 'train'
        else: split = 'val'
        kwargs_ = {'transform': kwargs['transform'],
                   'target_transform': kwargs['target_transform']}
        super(ImageNet, self).__init__(root, split, **kwargs_)


class ImageFolder(tv_datasets.ImageFolder):
    def __init__(self,
                 roots,
                 train=True,
                 transform=None,
                 target_transform=None,
                 *args, **kwargs):
        if train: root = roots[0]
        else: root = roots[1]
        super(ImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)


class STL10(tv_datasets.STL10):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        if train: split = 'train'
        else: split = 'test'
        super(STL10, self).__init__(root=root,
                                    split=split,
                                    folds=None,
                                    transform=transform, target_transform=target_transform,
                                    download=download)
        self.targets = self.labels


class Food101(tv_datasets.Food101):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        if train: split = 'train'
        else: split = 'test'
        super(Food101, self).__init__(root, split, transform, target_transform, download)
        self.targets = self._labels


class Caltech101(tv_datasets.Caltech101):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs):
        super(Caltech101, self).__init__(root, target_type='category', transform=transform, target_transform=target_transform, download=download)
        self.targets = self.y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if len(img.getbands()) == 1:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
