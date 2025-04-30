import os
import collections
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
import advertorch.attacks as attacks
from sklearn.model_selection import train_test_split

import datasets
import models
import utils


def init_dataset(cfg):
    # dataset transforms
    train_transforms = []
    val_transforms = []

    # dataset parameters
    if cfg.dataset.lower() == 'mnist':
        dataset = datasets.common.MNIST
        data_path = os.path.join(cfg.data_dir, 'mnist')
        cfg.input_dims = [1, 28, 28]
        cfg.standardize = [(0.1307,),
                           (0.3081,)]
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'cifar10':
        dataset = datasets.common.CIFAR10
        data_path = os.path.join(cfg.data_dir, 'cifar10')
        cfg.input_dims = [3, 32, 32]
        cfg.standardize = [(0.4914, 0.4822, 0.4465),
                           (0.2470, 0.2435, 0.2616)]
        if cfg.model_type.lower() == 'vit':
            train_transforms.append(transforms.RandomCrop(32, padding=4))
            train_transforms.append(transforms.RandomHorizontalFlip())
            train_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'cifar100':
        dataset = datasets.common.CIFAR100
        data_path = os.path.join(cfg.data_dir, 'cifar100')
        cfg.input_dims = [3, 32, 32]
        cfg.standardize = [(0.5071, 0.4866, 0.4409),
                           (0.2673, 0.2564, 0.2761)]
        if cfg.model_type.lower() == 'vit':
            train_transforms.append(transforms.RandomCrop(32, padding=4))
            train_transforms.append(transforms.RandomHorizontalFlip())
            train_transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    # this dataset must be downloaded from the ImageNet website https://www.image-net.org/challenges/LSVRC/2012/index.php
    elif (cfg.dataset.lower() == 'imagenet') or (cfg.dataset.lower() == 'imagenet100'):
        dataset = datasets.common.ImageNet
        data_path = os.path.join(cfg.data_dir, 'imagenet')
        cfg.input_dims = [3, None, None]
        cfg.standardize = [(0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225)]
        if cfg.model_type.lower() == 'vit':
            train_transforms.append(transforms.RandomResizedCrop(224))
            train_transforms.append(transforms.RandomHorizontalFlip())
            train_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
        else:
            train_transforms.append(transforms.Resize(256))
            train_transforms.append(transforms.CenterCrop(224))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.Resize(256))
        val_transforms.append(transforms.CenterCrop(224))
        val_transforms.append(transforms.ToTensor())
    # download from https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
    elif cfg.dataset.lower() == 'tinyimagenet':
        dataset = datasets.tinyimagenet.TinyImageNet
        utils.unzip_file(os.path.join(cfg.data_dir, 'tinyimagenet', 'tiny-imagenet-200.zip'), os.path.join(cfg.data_dir, 'tinyimagenet'))
        data_path = os.path.join(cfg.data_dir, 'tinyimagenet', 'tiny-imagenet-200')
        cfg.input_dims = [3, 64, 64]
        cfg.standardize = [(0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225)]
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'svhn':
        dataset = datasets.common.SVHN
        data_path = os.path.join(cfg.data_dir, 'svhn')
        cfg.input_dims = [3, 32, 32]
        cfg.standardize = [(0.4377, 0.4438, 0.4728),
                           (0.1980, 0.2010, 0.1970)]
        if cfg.model_type.lower() == 'vit':
            train_transforms.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
            train_transforms.append(transforms.RandomRotation(degrees=15))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'stl10':
        dataset = datasets.common.STL10
        data_path = os.path.join(cfg.data_dir, 'stl10')
        cfg.input_dims = [3, 96, 96]
        cfg.standardize = [(0.4467, 0.4398, 0.4066),
                           (0.2603, 0.2566, 0.2713)]
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'food101':
        dataset = datasets.common.Food101
        utils.untar_file(os.path.join(cfg.data_dir, 'food101', 'food-101.tar.gz'))
        data_path = os.path.join(cfg.data_dir, 'food101')
        cfg.input_dims = [3, 224, 224]
        cfg.standardize = [(0.5577, 0.4424, 0.3272,),
                           (0.2591, 0.2630, 0.2657,)]
        if cfg.model_type.lower() == 'vit':
            train_transforms.append(transforms.RandomResizedCrop(224))
            train_transforms.append(transforms.RandomHorizontalFlip())
            train_transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
        else:
            train_transforms.append(transforms.Resize(256))
            train_transforms.append(transforms.CenterCrop(224))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.Resize(256))
        val_transforms.append(transforms.CenterCrop(224))
        val_transforms.append(transforms.ToTensor())
    elif cfg.dataset.lower() == 'caltech101':
        dataset = datasets.common.Caltech101
        utils.untar_file(os.path.join(cfg.data_dir, 'caltech101', 'caltech101', '101_ObjectCategories.tar.gz'))
        utils.untar_file(os.path.join(cfg.data_dir, 'caltech101', 'caltech101', 'Annotations.tar'))
        data_path = os.path.join(cfg.data_dir, 'caltech101')
        cfg.input_dims = [3, 224, 224]
        cfg.standardize = [(0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225)]
        train_transforms.append(transforms.Resize(256))
        train_transforms.append(transforms.CenterCrop(224))
        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.Resize(256))
        val_transforms.append(transforms.CenterCrop(224))
        val_transforms.append(transforms.ToTensor())
    else:
        raise NotImplementedError()

    # dataloaders
    if cfg.dataset.lower() == 'caltech101':
        dataset_all = dataset(root=data_path,
                              transform=transforms.Compose(train_transforms),
                              target_transform=None,
                              download=False,
                              )
        train_indices, val_indices = train_test_split(list(range(len(dataset_all))), test_size=0.1, random_state=cfg.random_seed)
        dataset_train = datasets.utils.Subset(dataset_all, train_indices)
        dataset_val = datasets.utils.Subset(dataset_all, val_indices)
    elif cfg.dataset.lower() == 'imagenet100':
        dataset_train = dataset(root=data_path,
                                train=True,
                                transform=transforms.Compose(train_transforms),
                                target_transform=None,
                                download=True,
                                )
        dataset_val = dataset(root=data_path,
                              train=False,
                              transform=transforms.Compose(val_transforms),
                              target_transform=None,
                              download=True,
                              )

        all_train_class_indices = list(set(target for _, target in dataset_train.samples))
        all_train_class_indices.sort()

        selected_class_indices = random.sample(all_train_class_indices, 100)
        selected_class_indices_set = set(selected_class_indices)

        selected_train_sample_indices = [i for i, (_, label) in enumerate(dataset_train.samples) if label in selected_class_indices_set]
        selected_val_sample_indices = [i for i, (_, label) in enumerate(dataset_val.samples) if label in selected_class_indices_set]

        map_class = collections.defaultdict(int)
        for (i, class_idx) in enumerate(selected_class_indices):
            map_class[class_idx] = i

        dataset_train.target_transform = datasets.utils.TargetTransform(map_class)
        dataset_val.target_transform = datasets.utils.TargetTransform(map_class)

        dataset_train = datasets.utils.Subset(dataset_train, selected_train_sample_indices)
        dataset_val = datasets.utils.Subset(dataset_val, selected_val_sample_indices)
        utils.save_txt(all_train_class_indices, os.path.join(cfg.log_dir, cfg.stdout_dir, 'all_train_class_indices.txt'))
        utils.save_txt(selected_class_indices, os.path.join(cfg.log_dir, cfg.stdout_dir, 'selected_class_indices.txt'))
    else:
        dataset_train = dataset(root=data_path,
                                train=True,
                                transform=transforms.Compose(train_transforms),
                                target_transform=None,
                                download=True,
                                )
        dataset_val = dataset(root=data_path,
                              train=False,
                              transform=transforms.Compose(val_transforms),
                              target_transform=None,
                              download=True,
                              )

    dataset_train_targets = torch.as_tensor(dataset_train.targets)
    dataset_val_targets = torch.as_tensor(dataset_val.targets)

    sampler = None
    cfg.class_weights = None
    if cfg.weighted_sampler:
        class_count = torch.bincount(dataset_train_targets)
        class_weights = 1. / class_count.float()
        sample_weights = class_weights[dataset_train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    if cfg.criterion == 'ce':
        cfg.c_dim = len(torch.unique(dataset_train_targets)) # number of output classes (based only on classes seen in training data)
        print('num_training_classes:', cfg.c_dim)
    else:
        cfg.c_dim = 1

    mb_total = len(dataset_train_targets) // cfg.batch_size # total number of training minibatches
    print('num_training:', len(dataset_train_targets))
    print('num_training_minibatches:', mb_total)
    print('num_validation:', len(dataset_val_targets))
    utils.flush()

    return dataset_train, dataset_val, sampler


def init_model(device, cfg):
    # define model
    model_kwargs = {
                    'order': cfg.order,
                    'alpha': cfg.alpha,
                    'iterations': cfg.iterations,
                    'eps': cfg.eps,
                    'noise_train': cfg.noise_train,
                    }
    if cfg.model_type.lower() == 'fc':
        model = models.base_models.fc.FCNet(cfg.input_dims,
                                            cfg.c_dim,
                                            cfg.param,
                                            cfg.standardize,
                                            cfg=cfg,
                                            device=device,
                                            **model_kwargs).to(device)
    elif cfg.model_type.lower() == 'resnet':
        resnet_block = getattr(models.resnet, cfg.param[0][0])
        resnet_layers = cfg.param[1]
        width_per_group = cfg.param[2][0]
        model = models.resnet.ResNet(
                                    block=resnet_block,
                                    layers=resnet_layers,
                                    c_in=cfg.input_dims[0],
                                    num_classes=cfg.c_dim,
                                    standardize=cfg.standardize,
                                    norm_layer=cfg.norm,
                                    activation=cfg.activation.lower(),
                                    width_per_group=width_per_group,
                                    cfg=cfg,
                                    device=device,
                                    **model_kwargs).to(device)
    elif cfg.model_type.lower() == 'wideresnet':
        depth = cfg.param[0][0]
        widen_factor = cfg.param[1][0]
        dropout = cfg.param[2][0]
        model = models.wideresnet.WideResNet(
                                    depth=depth,
                                    widen_factor=widen_factor,
                                    dropout=dropout,
                                    c_in=cfg.input_dims[0],
                                    num_classes=cfg.c_dim,
                                    standardize=cfg.standardize,
                                    norm_layer=cfg.norm,
                                    activation=cfg.activation.lower(),
                                    cfg=cfg,
                                    device=device,
                                    **model_kwargs
                                    ).to(device)
    elif cfg.model_type.lower() == 'vit':
        patch_size = cfg.param[0][0]
        num_layers = cfg.param[0][1]
        num_heads = cfg.param[0][2]
        hidden_dim = cfg.param[1][0]
        mlp_dim = cfg.param[1][1]
        dropout = cfg.param[2][0]
        attention_dropout = cfg.param[2][1]
        image_size = cfg.param[3][0]
        model = models.vit.VisionTransformer(
                                    image_size=image_size,
                                    patch_size=patch_size,
                                    num_layers=num_layers,
                                    num_heads=num_heads,
                                    hidden_dim=hidden_dim,
                                    mlp_dim=mlp_dim,
                                    dropout=dropout,
                                    attention_dropout=attention_dropout,
                                    num_classes=cfg.c_dim,
                                    c_in=cfg.input_dims[0],
                                    standardize=cfg.standardize,
                                    norm_layer=cfg.norm,
                                    activation=cfg.activation.lower(),
                                    cfg=cfg,
                                    device=device,
                                    **model_kwargs
                                    ).to(device)
    else:
        raise NotImplementedError()

    return model


def init_optimizer(model, cfg):
    # optimizer & learning rate scheduler
    cfg.lr = cfg.optim_param[cfg.optim.lower()]['lr']
    if cfg.optim.lower() == 'sgd':
        optimizer = optim.SGD(params=model.parameters(),
                              lr=cfg.lr,
                              weight_decay=cfg.optim_param['sgd']['weight_decay'],
                              momentum=cfg.optim_param['sgd']['momentum'],
                              nesterov=cfg.optim_param['sgd']['nesterov'])
    elif cfg.optim.lower() == 'adam':
        optimizer = optim.Adam(params=model.parameters(),
                               lr=cfg.lr,
                               weight_decay=cfg.optim_param['adam']['weight_decay'],
                               betas=(cfg.optim_param['adam']['beta1'],
                                     cfg.optim_param['adam']['beta2']))
    elif cfg.optim.lower() == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(),
                                lr=cfg.lr,
                                weight_decay=cfg.optim_param['adamw']['weight_decay'],
                                betas=(cfg.optim_param['adamw']['beta1'],
                                       cfg.optim_param['adamw']['beta2']),
                                eps=cfg.optim_param['adamw']['eps_opt'])
    else:
        raise NotImplementedError()
    if (cfg.model_type.lower() == 'vit') and (cfg.dataset.lower() != 'svhn'):
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=10,
            )
        main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6,
            )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[10],
            )
    elif cfg.dataset.lower() in ('imagenet', 'tinyimagenet', 'food101', 'imagenet100', 'caltech101',):
        milestones = [40, 70, 90]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    return optimizer, scheduler


def init_adversary(model, cfg):
    # adversarial training/evaluation parameters
    adversary = None
    if cfg.adv_train or cfg.adv_eval:
        if cfg.adv_norm == 'inf': attack_model = attacks.LinfPGDAttack
        elif cfg.adv_norm == '2': attack_model = attacks.L2PGDAttack
        elif cfg.adv_norm == '1': attack_model = attacks.L1PGDAttack
        else: raise NotImplementedError()
        adversary = attack_model(model,
                                 loss_fn=nn.CrossEntropyLoss (reduction='mean'),
                                 eps=cfg.adv_eps,
                                 nb_iter=cfg.adv_nb_iter,
                                 eps_iter=cfg.adv_eps_iter,
                                 rand_init=True,
                                 clip_min=0.0,
                                 clip_max=1.0,
                                 targeted=False)
    return adversary
