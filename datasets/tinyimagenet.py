import sys
import os
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, '..')
import utils


class TinyImageNet(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 *args, **kwargs
                 ):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.targets = []

        self.map_files = {}
        self.map_class = {}
        wnids = utils.read_param(os.path.join(self.root, 'wnids.txt'))
        for (i, id_) in enumerate(wnids):
            self.map_class[id_[0]] = i

        counter = 0
        if train:
            dir_contents = os.listdir(os.path.join(self.root, 'train'))
            for folder in dir_contents:
                folder_files = os.listdir(os.path.join(self.root, 'train', folder, 'images'))
                for file in folder_files:
                    filepath = os.path.join(self.root, 'train', folder, 'images', file)
                    self.map_files[counter] = [filepath, self.map_class[folder]]
                    self.targets.append(self.map_class[folder])
                    counter += 1
        else:
            annotations = utils.read_param(os.path.join(self.root, 'val', 'val_annotations.txt'), delimeter='\t')
            for label in annotations:
                filepath = os.path.join(self.root, 'val', 'images', label[0])
                self.map_files[counter] = [filepath, self.map_class[label[1]]]
                self.targets.append(self.map_class[label[1]])
                counter += 1

    def __len__(self):
        return len(self.map_files)

    def __getitem__(self, index):
        filepath, label = self.map_files[index]

        image = Image.open(filepath)

        if len(image.getbands()) == 1:
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
