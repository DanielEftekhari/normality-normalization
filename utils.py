import sys
import os
import pathlib
import shutil
import zipfile
import tarfile
import csv
import json
import yaml
import h5py
import pickle
import datetime

import math
import numpy as np
import torch


class Meter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.
        self.var = 0.
        self.std = 0.
        self.count = 0

    def update(self, vals, m):
        if not isinstance(vals, torch.Tensor):
            vals = torch.tensor(vals)

        sums = torch.sum(vals).item()
        squared_sums = torch.sum(torch.square(vals)).item()

        # compute the new mean & variance
        new_avg = (self.avg * self.count + sums) / (self.count + m)
        new_var = self.count / (self.count + m) * (self.var + math.pow(self.avg, 2)) + 1. / (self.count + m) * squared_sums - math.pow(new_avg, 2)

        # make updates
        self.avg = new_avg
        self.var = max(0., new_var) # enforces numerical stability for variances close to 0
        self.std = math.sqrt(self.var)
        self.count += m


class TensorMeter():
    def __init__(self, c=1, device=None):
        self.c = c
        self.device = device
        self.reset()

    def reset(self):
        self.avg = torch.zeros(size=(self.c,)).to(self.device)
        self.var = torch.zeros(size=(self.c,)).to(self.device)
        self.std = torch.zeros(size=(self.c,)).to(self.device)
        self.count = 0

    def update(self, vals):
        if not isinstance(vals, torch.Tensor):
            vals = torch.tensor(vals).to(self.device)

        sums = vals
        squared_sums = torch.square(vals)

        # compute the new mean & variance
        new_avg = (self.avg * self.count + sums) / (self.count + 1)
        new_var = self.count / (self.count + 1) * (self.var + torch.square(self.avg)) + 1. / (self.count + 1) * squared_sums - torch.square(new_avg)

        # make updates
        self.avg = new_avg
        self.var = torch.maximum(new_var, torch.zeros_like(new_var))
        self.std = torch.sqrt(self.var)
        self.count += 1


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def bootstrap_ci(data, num_sample=1000, alpha=0.025):
    if type(data) == list: data = np.array(data)
    assert (len(data.shape) == 1)
    n = data.shape[0]
    x_bar = np.mean(data)

    samples = np.random.choice(data, size=(num_sample, n), replace=True)
    delta = np.mean(samples, axis=1) - x_bar
    delta = np.sort(delta)

    l, r = delta[int((num_sample-1) * (1.-alpha))], delta[int((num_sample-1) * alpha)]
    return x_bar-l, x_bar, x_bar-r


def tensor2array(tensor):
    return tensor.cpu().detach().clone().numpy()


def get_class_outputs(outputs):
    return torch.argmax(outputs, dim=-1)


def logits_to_probs(logits):
    c = torch.max(logits, dim=-1)[0]
    return torch.softmax(logits - c[:, None], dim=-1)


def to_one_hot(y, c_dim):
    y_one_hot = torch.zeros(size=(y.shape[0], c_dim)).to(y.device)
    y_one_hot.scatter_(1, y.unsqueeze(-1), 1)
    return y_one_hot


def where(condition, x, y=0.):
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y).to(x.device)
    return torch.where(condition, x, y)


def can_be_float(x):
    try:
        _ = float(x)
        return True
    except:
        return False


def a_better_than_b(a, b, comparator='max'):
    if comparator == 'max':
        return a > b
    else:
        return a < b


def make_dirs(path, replace=False):
    if path is None:
        return
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if replace:
            rm_dir(path)
            os.makedirs(path)


def read_param(filepath, delimeter=' '):
    if filepath is None or (not os.path.isfile(filepath)):
        return None

    res = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split(delimeter)

            line_formatted = []
            for elem in line:
                if elem.endswith('\n'):
                    elem = elem.split('\n')[0]

                if elem.isnumeric():
                    line_formatted.append(int(elem))
                elif can_be_float(elem):
                    line_formatted.append(float(elem))
                else:
                    line_formatted.append(elem)
            res.append(line_formatted)

    return res


def load_json(filepath):
    if filepath is None or (not os.path.isfile(filepath)):
        return None
    with open(filepath) as data_file:
        data = json.load(data_file)
    return data


def init_json(filepath):
    if filepath is None:
        return
    filepath = add_extension(filepath, '.json')
    if not os.path.isfile(filepath):
        save_json({}, filepath)


def save_json(data, filepath, sort_keys=False):
    if filepath is None:
        raise ValueError()
    filepath = add_extension(filepath, '.json')
    with open(filepath, 'w') as outfile:
        json.dump(obj=data, fp=outfile, sort_keys=sort_keys, indent=4, separators=(',', ': '))


def load_yaml(filepath):
    if filepath is None or (not os.path.isfile(filepath)):
        return None
    with open(filepath) as data_file:
        data = yaml.safe_load(data_file)
    return data


def save_yaml(data, filepath):
    if filepath is None:
        raise ValueError()
    filepath = add_extension(filepath, '.yaml')
    with open(filepath, 'w') as outfile:
        yaml.dump(data, outfile, sort_keys=True, indent=4)


def save_txt(data, filepath, delimeter=','):
    if filepath is None:
        raise ValueError()
    with open(filepath, 'w', newline='\n') as file:
        for (i, line) in enumerate(data):
            if type(line) in (tuple, list):
                file.writelines('{}\n'.format(delimeter.join(line)))
            else:
                file.writelines('{}\n'.format(line))


def save_array(data, filepath):
    if filepath is None:
        raise ValueError()
    filepath = add_extension(filepath, '.npy')
    np.save(file=filepath, arr=data)


def init_hdf5(filepath, mode='a'):
    if filepath is None:
        return
    filepath = add_extension(filepath, '.hdf5')
    if mode == 'r' and (not os.path.isfile(filepath)):
        return None
    f = h5py.File(filepath, mode)
    return f


def save_hdf5(f, name, data=None, dtype=None):
    if name in f:
        del f[name]
    f.create_dataset(name, data=data, dtype=dtype)


def close_hdf5(f):
    f.close()


def init_save_close_hdf5(filepath, name, mode='a', data=None, dtype=None):
    if filepath is None:
        raise ValueError()
    f = init_hdf5(filepath, mode=mode)
    save_hdf5(f, name, data=data, dtype=dtype)
    close_hdf5(f)


def load_pkl(filepath):
    if not os.path.isfile(filepath):
        return None
    data = pickle.load(open(filepath, 'rb'))
    return data


def save_pkl(filepath, data):
    if filepath is None:
        raise ValueError()
    filepath = add_extension(filepath, '.pkl')
    pickle.dump(data, open(filepath, 'wb'))


def add_extension(filepath, extension):
    if not filepath.lower().endswith(extension):
        filepath = '{}{}'.format(filepath, extension)
    return filepath


def rm_file(filepath):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return
    os.remove(filepath)


def rm_dir(dirpath):
    if (dirpath is None) or (not os.path.exists(dirpath)):
        return
    shutil.rmtree(dirpath)


def unzip_file(filepath, destdir):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(destdir)


def untar_file(filepath):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return
    tar = tarfile.open(filepath)
    parent_dirpath = str(pathlib.Path(filepath).parent.absolute())
    tar.extractall(path=parent_dirpath)
    tar.close()


def tar_dir(dirpath):
    if (dirpath is None) or (not os.path.exists(dirpath)):
        return
    parent_dirpath = str(pathlib.Path(dirpath).parent.absolute())
    basedir_name = os.path.basename(dirpath)
    filepath = os.path.join(parent_dirpath, basedir_name)
    shutil.make_archive(filepath, 'tar', dirpath)
    return '{}.tar'.format(filepath)


def copy_file(filepath, targetpath):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return
    shutil.copyfile(filepath, targetpath)


def copy_folder(dirpath, targetdir):
    if (dirpath is None) or (not os.path.exists(dirpath)):
        return
    shutil.copy(dirpath, targetdir)


def move_file(filepath, targetpath, copy_move=False):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return

    if copy_move:
        # <copy-and-delete> rather than <move>, in case source and destination are on separate filesystems
        shutil.copyfile(filepath, targetpath)
        os.remove(filepath)
    else:
        shutil.move(filepath, targetpath)


def move_folder(dirpath, targetdir):
    if (dirpath is None) or (not os.path.exists(dirpath)):
        return
    # shutil.move(dirpath, targetdir)
    shutil.copy(dirpath, targetdir) # need to <copy-and-delete> rather than <move>, in case source and destination are on separate filesystems
    shutil.rmtree(dirpath)


def load_csv(filepath, mode='list'):
    if (filepath is None) or (not os.path.isfile(filepath)):
        return

    res = []
    with open(filepath, 'r') as file:
        if mode == 'list':
            reader = csv.reader(file)
        elif mode == 'dict':
            reader = csv.DictReader(file)
        else:
            raise ValueError()

        for line in reader:
            res.append(line)

    return res


def get_current_time():
    return str(datetime.datetime.utcnow()).replace(':', '-').replace(' ', '-').replace('.', '-')


def flush():
    print('\n')
    sys.stdout.flush()
