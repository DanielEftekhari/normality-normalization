import sys
import os
import collections
import argparse
import time
import pprint

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from advertorch.context import ctx_noparamgrad_and_eval

import common
import metrics
import configs.config as config
import configs.utils as cutils
import plotting.utils as putils
import utils


class Trainer():
    def __init__(self, dataloader_train, dataloader_val, device, cfg):
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        self.cfg = cfg
        self.device = device

        # number of running parameter estimates to plot (for each applicable layer)
        self._num_plot_running_stats = 5
        self._log_freq = 5
        self._progress_freq = 25

        # loss function
        if self.cfg.criterion == 'ce':
            self.measure, self.perf_sign = 'acc', 'max'
            self._measure_name = 'Accuracy'
            self.criterion = nn.CrossEntropyLoss(reduction='none',
                                                 weight=self.cfg.class_weights,
                                                 )
        else:
            if self.cfg.criterion == 'l2':
                self.measure = 'l2_avg'
                self._measure_name = 'Mean Squared Error'
                self.criterion = nn.MSELoss(reduction='none')
            elif self.cfg.criterion == 'l1':
                self.measure = 'l1_avg'
                self._measure_name = 'Mean Absolute Error'
                self.criterion = nn.L1Loss(reduction='none')
            self.perf_sign = 'min'
            self.criterionl2 = nn.MSELoss(reduction='none')
            self.criterionl1 = nn.L1Loss(reduction='none')

        if (self.cfg.model_type.lower() == 'vit') and (self.cfg.dataset.lower() in ('cifar10', 'cifar100', 'food101')):
            self.mixup = v2.MixUp(num_classes=self.cfg.c_dim)
        else:
            self.mixup = None

    def train(self, model, optimizer, scheduler, adversary):
        # track training and validation stats over epochs
        self.metrics = collections.defaultdict(list)

        # best model is defined as
        # a) model with highest accuracy (if a classification task), else b) model with lowest regression error
        # load model checkpoint if it exists
        last_model = self._load_model(model, optimizer, scheduler)
        if last_model is None:
            loaded_model = False
            curr_epoch = 0
            if self.measure == 'acc': self.best = 0.
            else: self.best = float('inf')
        else:
            loaded_model = True
            curr_epoch, self.best, perf = last_model
            self._load_prev_metrics()
            print('loaded previously saved model:'
                  'epoch{:0=4d}, performance{:.4f}, running best{:.4f}'.format(curr_epoch, perf, self.best))
            print('loaded scheduler, learning rate is set to:',
                  scheduler.get_last_lr()[0])
            utils.flush()

        # measure performance before any training is done
        if not loaded_model:
            self.metrics['epochs'].append(curr_epoch)

            matrix_train = self.validate(model, adversary, self.dataloader_train, 'train')
            matrix_val = self.validate(model, adversary, self.dataloader_val, 'val')
            if self.cfg.adv_eval: matrix_val_adv = self.validate(model, adversary, self.dataloader_val, 'val_adv', eval_adv=True)

            if self.measure == 'acc':
                utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                        name='confusion_matrix_train_epoch{:0=4d}'.format(curr_epoch),
                                        data=utils.tensor2array(matrix_train), dtype=np.float32)
                utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                        name='confusion_matrix_val_epoch{:0=4d}'.format(curr_epoch),
                                        data=utils.tensor2array(matrix_val), dtype=np.float32)
                if self.cfg.adv_eval:
                    utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                            name='confusion_matrix_val_adv_epoch{:0=4d}'.format(curr_epoch),
                                            data=utils.tensor2array(matrix_val_adv), dtype=np.float32)

            if self.cfg.plot:
                for sublayer in model.named_buffers():
                    sublayer_name = sublayer[0]
                    sublayer_tensor = sublayer[1]
                    if 'running' in sublayer_name:
                        for j in range(0, self._num_plot_running_stats):
                            self.metrics['{}_{}'.format(sublayer_name, j)].append(sublayer_tensor[j].item())

            # save (initial) weights
            self._eval_best_model(curr_epoch)
            save_name, model_filepath = self._save_model(model, optimizer, scheduler, curr_epoch)
            if save_name is not None:
                self._pop_ckpt(save_name)
                self._model_tape.append(model_filepath)
                tar_filepath = utils.tar_dir(self.cfg.tmp_model_folder)
                if tar_filepath is not None:
                    utils.move_file(tar_filepath, os.path.join(self.cfg.model_folder, 'ckpt_temp.tar'), copy_move=True)
                    utils.move_file(os.path.join(self.cfg.model_folder, 'ckpt.tar'), os.path.join(self.cfg.model_folder, 'ckpt_backup.tar'), copy_move=False)
                    utils.move_file(os.path.join(self.cfg.model_folder, 'ckpt_temp.tar'), os.path.join(self.cfg.model_folder, 'ckpt.tar'), copy_move=False)
                    utils.rm_file(os.path.join(self.cfg.model_folder, 'ckpt_backup.tar'))
                with open(os.path.join(self.cfg.model_folder, 'STAGE.txt'), 'w') as file:
                    file.write('{}.pth'.format(save_name))
                utils.save_pkl(os.path.join(self.cfg.model_folder, 'model_tape.pkl'), self._model_tape)

            utils.copy_file(self.cfg.tmp_metric_path, os.path.join(self.cfg.metric_folder, 'metrics_temp.hdf5'))
            utils.move_file(self.cfg.metric_path, os.path.join(self.cfg.metric_folder, 'metrics_backup.hdf5'), copy_move=True)
            utils.move_file(os.path.join(self.cfg.metric_folder, 'metrics_temp.hdf5'), self.cfg.metric_path, copy_move=True)
            utils.rm_file(os.path.join(self.cfg.metric_folder, 'metrics_backup.hdf5'))
            tar_filepath = utils.tar_dir(self.cfg.tmp_plot_folder)
            if tar_filepath is not None:
                utils.move_file(tar_filepath, os.path.join(self.cfg.plot_folder, 'plots_temp.tar'), copy_move=True)
                utils.move_file(os.path.join(self.cfg.plot_folder, 'plots_temp.tar'), os.path.join(self.cfg.plot_folder, 'plots.tar'), copy_move=False)

        utils.copy_file(self.cfg.tmp_stdout_path, self.cfg.stdout_path[0:-4] + '_temp.txt')
        utils.move_file(self.cfg.stdout_path[0:-4] + '_temp.txt', self.cfg.stdout_path, copy_move=True)

        for epoch in range(curr_epoch+1, self.cfg.epochs+1):
            trainval_start_time = time.time()
            self.metrics['epochs'].append(epoch)

            # training
            if self.cfg.adv_train:
                self.train_one_epoch(model, optimizer, adversary, self.dataloader_train, train_adv=True)
            else:
                self.train_one_epoch(model, optimizer, adversary, self.dataloader_train)

            # validation
            matrix_train = self.validate(model, adversary, self.dataloader_train, 'train')
            matrix_val = self.validate(model, adversary, self.dataloader_val, 'val')
            if self.cfg.adv_eval:
                matrix_val_adv = self.validate(model, adversary, self.dataloader_val, 'val_adv', eval_adv=True)
            scheduler.step()

            if self.cfg.plot: # plotting
                for sublayer in model.named_buffers():
                    sublayer_name = sublayer[0]
                    sublayer_tensor = sublayer[1]
                    if 'running' in sublayer_name:
                        for j in range(0, self._num_plot_running_stats):
                            self.metrics['{}_{}'.format(sublayer_name, j)].append(sublayer_tensor[j].item())
                        putils.plot_line(self.metrics['epochs'],
                                           [self.metrics['{}_{}'.format(sublayer_name, j)] for j in range(self._num_plot_running_stats)],
                                            None,
                                            ['unit{}'.format(j) for j in range(self._num_plot_running_stats)],
                                            'Epoch Number', '{}'.format(sublayer_name), self.cfg)

                if self.cfg.adv_train:
                    putils.plot_line(self.metrics['epochs'],
                                       [self.metrics['train_adv_loss_avg'], self.metrics['val_loss_avg']],
                                       [self.metrics['train_adv_loss_std'], self.metrics['val_loss_std']],
                                       ['Training Set (Adversarial)', 'Validation Set'],
                                       'Epoch Number', 'Loss', self.cfg)
                    putils.plot_line(self.metrics['epochs'],
                                       [self.metrics['train_adv_{}'.format(self.measure)], self.metrics['val_{}'.format(self.measure)]],
                                       None,
                                       ['Training Set (Adversarial)', 'Validation Set'],
                                       'Epoch Number', self._measure_name, self.cfg)
                else:
                    putils.plot_line(self.metrics['epochs'],
                                       [self.metrics['train_loss_avg'], self.metrics['val_loss_avg']],
                                       [self.metrics['train_loss_std'], self.metrics['val_loss_std']],
                                       ['Training Set', 'Validation Set'],
                                       'Epoch Number', 'Loss', self.cfg)
                    putils.plot_line(self.metrics['epochs'],
                                       [self.metrics['train_{}'.format(self.measure)], self.metrics['val_{}'.format(self.measure)]],
                                       None,
                                       ['Training Set', 'Validation Set'],
                                       'Epoch Number', self._measure_name, self.cfg)
                if self.cfg.adv_eval:
                    putils.plot_line(self.metrics['epochs'],
                                       [self.metrics['val_adv_{}'.format(self.measure)]],
                                       None,
                                       ['Validation Set (Adversarial)'],
                                       'Epoch Number', self._measure_name, self.cfg)

            # metrics
            for metric in self.metrics:
                if 'confusion_matrix' not in metric:
                    utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                               name=metric,
                                               data=self.metrics[metric])
            if self.measure == 'acc':
                utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                           name='confusion_matrix_train_epoch{:0=4d}'.format(epoch),
                                           data=utils.tensor2array(matrix_train), dtype=np.float32)
                utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                           name='confusion_matrix_val_epoch{:0=4d}'.format(epoch),
                                           data=utils.tensor2array(matrix_val), dtype=np.float32)
                if self.cfg.adv_eval:
                    utils.init_save_close_hdf5(filepath=self.cfg.tmp_metric_path,
                                               name='confusion_matrix_val_adv_epoch{:0=4d}'.format(epoch),
                                               data=utils.tensor2array(matrix_val_adv), dtype=np.float32)

            trainval_end_time = time.time()
            print('train & val time', trainval_end_time-trainval_start_time)

            # model evaluation
            self._eval_best_model(epoch)
            save_name, model_filepath = self._save_model(model, optimizer, scheduler, epoch)

            file_handling_start_time = time.time()
            if save_name is not None:
                self._pop_ckpt(save_name)
                self._model_tape.append(model_filepath)
                tar_filepath = utils.tar_dir(self.cfg.tmp_model_folder)
                if tar_filepath is not None:
                    utils.move_file(tar_filepath, os.path.join(self.cfg.model_folder, 'ckpt_temp.tar'), copy_move=True)
                    utils.move_file(os.path.join(self.cfg.model_folder, 'ckpt.tar'), os.path.join(self.cfg.model_folder, 'ckpt_backup.tar'), copy_move=False)
                    utils.move_file(os.path.join(self.cfg.model_folder, 'ckpt_temp.tar'), os.path.join(self.cfg.model_folder, 'ckpt.tar'), copy_move=False)
                    utils.rm_file(os.path.join(self.cfg.model_folder, 'ckpt_backup.tar'))
                with open(os.path.join(self.cfg.model_folder, 'STAGE.txt'), 'w') as file:
                    file.write('{}.pth'.format(save_name))
                utils.save_pkl(os.path.join(self.cfg.model_folder, 'model_tape.pkl'), self._model_tape)

            utils.copy_file(self.cfg.tmp_stdout_path, self.cfg.stdout_path[0:-4] + '_temp.txt')
            utils.move_file(self.cfg.stdout_path[0:-4] + '_temp.txt', self.cfg.stdout_path, copy_move=True)
            utils.copy_file(self.cfg.tmp_metric_path, os.path.join(self.cfg.metric_folder, 'metrics_temp.hdf5'))
            utils.move_file(self.cfg.metric_path, os.path.join(self.cfg.metric_folder, 'metrics_backup.hdf5'), copy_move=True)
            utils.move_file(os.path.join(self.cfg.metric_folder, 'metrics_temp.hdf5'), self.cfg.metric_path, copy_move=True)
            utils.rm_file(os.path.join(self.cfg.metric_folder, 'metrics_backup.hdf5'))
            tar_filepath = utils.tar_dir(self.cfg.tmp_plot_folder)
            if tar_filepath is not None:
                utils.move_file(tar_filepath, os.path.join(self.cfg.plot_folder, 'plots_temp.tar'), copy_move=True)
                utils.move_file(os.path.join(self.cfg.plot_folder, 'plots_temp.tar'), os.path.join(self.cfg.plot_folder, 'plots.tar'), copy_move=False)
            file_handling_end_time = time.time()
            print('file handling time', file_handling_end_time-file_handling_start_time)
            utils.flush()

        # logging at training completion
        with open(os.path.join(self.cfg.model_folder, 'DONE.txt'), 'w') as file:
            file.write(str(self.cfg.epochs))

    def train_one_epoch(self, model, optimizer, adversary, dataloader, train_adv=False):
        model.train()

        for mb, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if self.cfg.criterion != 'ce': y = y.unsqueeze(1)

            if self.mixup is not None:
                x, y = self.mixup(x, y)

            if train_adv:
                with ctx_noparamgrad_and_eval(model):
                    x = adversary.perturb(x, y)
            optimizer.zero_grad()
            logits = model(x)
            losses = self.criterion(logits, y)
            torch.mean(losses).backward()
            optimizer.step()

    def validate(self, model, adversary, dataloader, prefix, eval_adv=False):
        model.eval()

        self.metrics_epoch = collections.defaultdict(utils.Meter)
        matrix = torch.zeros((self.cfg.c_dim, self.cfg.c_dim), dtype=torch.int32, device=self.device)
        for mb, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if self.cfg.criterion != 'ce': y = y.unsqueeze(1)

            if eval_adv:
                x = adversary.perturb(x, y)
            with torch.no_grad():
                logits = model(x)
            losses = self.criterion(logits, y)

            self.metrics_epoch['{}_loss'.format(prefix)].update(losses, x.shape[0])
            if self.measure == 'acc':
                cm = metrics.confusion_matrix(utils.get_class_outputs(logits), y, self.cfg.c_dim)
                matrix = matrix + cm
                acc1, acc5 = metrics.accuracy_topk(logits, y, topk=(1, 5))
                self.metrics_epoch['{}_acc1'.format(prefix)].update(acc1[0] * x.shape[0], x.shape[0])
                self.metrics_epoch['{}_acc5'.format(prefix)].update(acc5[0] * x.shape[0], x.shape[0])
            else:
                l1 = self.criterionl1(logits, y)
                l2 = self.criterionl2(logits, y)
                self.metrics_epoch['{}_l1'.format(prefix)].update(l1, x.shape[0])
                self.metrics_epoch['{}_l2'.format(prefix)].update(l2, x.shape[0])

        self._summarize_metrics(matrix, prefix)
        return matrix

    def _summarize_metrics(self, matrix, prefix):
        for key in sorted(self.metrics_epoch.keys()):
            self.metrics['{}_{}'.format(key, 'avg')].append(self.metrics_epoch[key].avg)
            self.metrics['{}_{}'.format(key, 'std')].append(self.metrics_epoch[key].std)

            print('epoch{:0=4d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'avg'), self.metrics['{}_{}'.format(key, 'avg')][-1]))
            print('epoch{:0=4d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_{}'.format(key, 'std'), self.metrics['{}_{}'.format(key, 'std')][-1]))

        if self.measure == 'acc':
            print(matrix)
            self.metrics['{}_acc'.format(prefix)].append(metrics.accuracy(matrix))
            print('epoch{:0=4d}_{}{:.4f}'.format(self.metrics['epochs'][-1], '{}_acc'.format(prefix), self.metrics['{}_acc'.format(prefix)][-1]))
        utils.flush()

    def _eval_best_model(self, epoch):
        if utils.a_better_than_b(self.metrics['val_{}'.format(self.measure)][-1], self.best, comparator=self.perf_sign):
            self.best = self.metrics['val_{}'.format(self.measure)][-1]
            print('new best model at epoch {:0=4d} with val_{} {:.4f} val_loss {:.4f}'.format(epoch, self.measure, self.best, self.metrics['val_loss_avg'][-1]))
            utils.flush()

    def _load_model(self, model, optimizer, scheduler):
        if not os.path.isfile(os.path.join(self.cfg.model_folder, 'STAGE.txt')):
            self._model_tape = collections.deque()
            return None

        with open(os.path.join(self.cfg.model_folder, 'STAGE.txt'), 'r') as file:
            line = file.readline()

        utils.copy_file(os.path.join(self.cfg.model_folder, 'ckpt.tar'), os.path.join(self.cfg.tmp_model_folder, 'ckpt.tar'))
        utils.untar_file(os.path.join(self.cfg.tmp_model_folder, 'ckpt.tar'))
        checkpoint = torch.load(os.path.join(self.cfg.tmp_model_folder, line))
        utils.rm_file(os.path.join(self.cfg.tmp_model_folder, 'ckpt.tar'))

        model_tape = utils.load_pkl(os.path.join(self.cfg.model_folder, 'model_tape.pkl'))
        self._model_tape = collections.deque(model_tape)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        best = checkpoint['best']
        perf = checkpoint['perf']
        return epoch, best, perf

    def _save_model(self, model, optimizer, scheduler, epoch):
        if self.cfg.save_model in ('best', 'all'):
            save_name = '{}_epoch{:0=4d}_val_{}{:.4f}'.format(self.cfg.model_name, epoch, self.measure, self.metrics['val_{}'.format(self.measure)][-1])
            model_filepath = os.path.join(self.cfg.tmp_model_folder, '{}.pth'.format(save_name))
            torch.save({
                       'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(),
                       'val_{}'.format(self.measure): self.metrics['val_{}'.format(self.measure)][-1],
                       'epoch': epoch,
                       'best': self.best,
                       'perf': self.metrics['val_{}'.format(self.measure)][-1],
                       }, model_filepath)

            return save_name, model_filepath
        return None, None

    def _pop_ckpt(self, save_name):
        if self.best == self.metrics['val_{}'.format(self.measure)][-1]:
            with open(os.path.join(self.cfg.model_folder, 'best_{}.txt'.format(self.cfg.model_name)), 'w') as file:
                file.write('{}.pth'.format(save_name))

            if self.cfg.save_model == 'best':
                while len(self._model_tape) > 0:
                    filepath_to_rm = self._model_tape.popleft()
                    utils.rm_file(filepath_to_rm)
        else:
            if self.cfg.save_model == 'best':
                while len(self._model_tape) > 1:
                    filepath_to_rm = self._model_tape.pop()
                    utils.rm_file(filepath_to_rm)

    def _load_prev_metrics(self):
        if os.path.isfile(self.cfg.metric_path):
            prev_metrics_f = utils.init_hdf5(self.cfg.metric_path, mode='r')
            for name in prev_metrics_f:
                self.metrics[name] = prev_metrics_f[name][...].tolist()
            utils.close_hdf5(prev_metrics_f)


def init_dataloader(dataset_train, dataset_val, cfg, sampler=None):
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=cfg.batch_size,
                                  shuffle=cfg.shuffle if sampler is None else False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  sampler=sampler,
                                  )

    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                pin_memory=True,
                                drop_last=False,
                                )

    return dataloader_train, dataloader_val


def setup(cfg):
    base_cfg = utils.load_yaml(cfg.base_config_path)
    # override base-config parameters with arguments provided at runtime
    if cfg.custom_config_path is not None: custom_cfg = utils.load_yaml(cfg.custom_config_path)
    else: custom_cfg = None
    membership = cutils.get_membership(base_cfg)
    cfg_dict = vars(cfg)
    cfg_dict = {key: cfg_dict[key] for key in cfg_dict if cfg_dict[key] is not None}

    updated_cfg = base_cfg
    if custom_cfg is not None:
        updated_cfg = cutils.update_params(updated_cfg, custom_cfg, membership)
    updated_cfg = cutils.update_params(updated_cfg, cfg_dict, membership)

    if updated_cfg['param'] is None:
        # define network architecture (if not set as a command-line option)
        updated_cfg['param'] = utils.read_param(os.path.join(updated_cfg['param_dir'], '{}_param.txt'.format(updated_cfg['model_type'].lower())))

    if updated_cfg['debug_mode']:
        torch.autograd.set_detect_anomaly(True)

    # set random seed
    if updated_cfg['random_seed'] == -1: # if not already set
        updated_cfg['random_seed'] = random.randint(1, int(1e5))
        print('random seed set to {}'.format(updated_cfg['random_seed']))
    random.seed(updated_cfg['random_seed'])
    np.random.seed(updated_cfg['random_seed'])
    torch.manual_seed(updated_cfg['random_seed'])

    # set device as cuda or cpu
    if updated_cfg['device'].lower() == 'cuda' and torch.cuda.is_available():
        # reproducibility using cuda
        torch.cuda.manual_seed(updated_cfg['random_seed'])
        cudnn.deterministic = True
        cudnn.benchmark = True
    elif updated_cfg['device'].lower() == 'cuda':
        print('device option was set to <cuda>, but no cuda device was found; reverting device to <cpu>')
        updated_cfg['device'] = 'cpu'
        utils.flush()

    start_time = utils.get_current_time()
    updated_cfg['start_time'] = start_time
    if updated_cfg['model_name'] is None: updated_cfg['model_name'] = start_time
    updated_cfg['log_dir'] = os.path.join(updated_cfg['log_dir'], updated_cfg['model_name'])
    updated_cfg['tmp_log_dir'] = os.path.join(updated_cfg['tmp_log_dir'], updated_cfg['model_name'])

    cfg = argparse.Namespace(**updated_cfg)

    utils.make_dirs(os.path.join(cfg.log_dir, cfg.config_dir), replace=False)
    utils.make_dirs(cfg.tmp_log_dir, replace=False)
    utils.save_yaml(updated_cfg, '{}/config.yaml'.format(os.path.join(cfg.log_dir, cfg.config_dir)))

    # writing to stdout
    utils.make_dirs(os.path.join(cfg.log_dir, cfg.stdout_dir), replace=False)
    utils.make_dirs(os.path.join(cfg.tmp_log_dir, cfg.stdout_dir), replace=False)

    stdout_name_template = 'stdout_{}_{:0=4d}.txt'
    stdout_path_template = os.path.join(cfg.log_dir, cfg.stdout_dir, stdout_name_template)
    stdout_count = 1
    while os.path.isfile(stdout_path_template.format(cfg.model_name, stdout_count)):
        stdout_count += 1
    stdout_name = stdout_name_template.format(cfg.model_name, stdout_count)

    cfg.stdout_path = os.path.join(cfg.log_dir, cfg.stdout_dir, stdout_name)
    cfg.tmp_stdout_path = os.path.join(cfg.tmp_log_dir, cfg.stdout_dir, stdout_name)
    sys.stdout = open(cfg.tmp_stdout_path, 'a')
    print('start_time', start_time)
    pprint.pprint(cfg.__dict__)
    utils.flush()

    # setting up output directories
    cfg.metric_folder = os.path.join(cfg.log_dir, cfg.metric_dir)
    cfg.tmp_metric_folder = os.path.join(cfg.tmp_log_dir, cfg.metric_dir)
    utils.make_dirs(cfg.metric_folder, replace=False)
    utils.make_dirs(cfg.tmp_metric_folder, replace=False)
    cfg.tmp_metric_path = os.path.join(cfg.tmp_metric_folder, 'metrics.hdf5')
    cfg.metric_path = os.path.join(cfg.metric_folder, 'metrics.hdf5')

    cfg.model_folder = os.path.join(cfg.log_dir, cfg.model_dir)
    cfg.tmp_model_folder = os.path.join(cfg.tmp_log_dir, cfg.model_dir)
    utils.make_dirs(cfg.model_folder, replace=False)
    utils.make_dirs(cfg.tmp_model_folder, replace=False)

    if cfg.plot:
        cfg.plot_folder = os.path.join(cfg.log_dir, cfg.plot_dir)
        cfg.tmp_plot_folder = os.path.join(cfg.tmp_log_dir, cfg.plot_dir)
        utils.make_dirs(cfg.plot_folder, replace=False)
        utils.make_dirs(cfg.tmp_plot_folder, replace=False)
    else:
        cfg.plot_folder = None
        cfg.tmp_plot_folder = None

    return cfg


def main(cfg):
    device = torch.device(cfg.device)

    dataset_train, dataset_val, sampler = common.init_dataset(cfg)
    dataloader_train, dataloader_val = init_dataloader(dataset_train, dataset_val, cfg, sampler=sampler)
    model = common.init_model(device, cfg)
    optimizer, scheduler = common.init_optimizer(model, cfg)
    adversary = common.init_adversary(model, cfg)

    # model training
    trainer = Trainer(dataloader_train, dataloader_val, device, cfg)
    trainer.train(model, optimizer, scheduler, adversary)
    return 0


if __name__ == '__main__':
    stdout_orig = sys.__stdout__

    parser = config.training()
    cfg, _ = parser.parse_known_args()
    cfg = setup(cfg)

    exit_status = main(cfg)

    # clean-up temporary directory
    try:
        sys.stdout = stdout_orig
        utils.rm_dir(cfg.tmp_log_dir)
    except:
        print('could not remove temporary directory {}'.format(cfg.tmp_log_dir))
    print('exit status', exit_status)
