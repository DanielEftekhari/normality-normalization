# Normality Normalization #

Code accompanying the ICML 2025 paper [On the Importance of Gaussianizing Representations](https://arxiv.org/abs/2505.00685).

Citation:
```
@misc{eftekhari2025importancegaussianizingrepresentations,
      title={On the Importance of Gaussianizing Representations}, 
      author={Daniel Eftekhari and Vardan Papyan},
      year={2025},
      eprint={2505.00685},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.00685}, 
}
```

Correspondences to: Daniel Eftekhari<br/>
defte@cs.toronto.edu

## Requirements:

Python 3.8 or higher.<br/>

Specific module and version requirements are listed in requirements.txt. After cloning the repository, `cd` to the repo directory, and enter `pip install -r requirements.txt`.

## File Descriptions:

Executable files:<br/>
`train.py`: model training<br/>

Configuration Specification files:<br/>
`./configs/config.py`: Complete list of command-line parsable arguments.<br/>
`./configs/default_config.yaml`: Default config values; arguments provided in the command line override the default values. Each model type also has its own default configuration file, located at `./configs/model_default_configs/<model_type>.yaml`.<br/>

Architecture Specification files:<br/>
`./params/resnet.txt`: ResNet architectures can be specified here. The order of arguments is: {block type, [layers_1, layers_2, layers_3, layers_4], width_per_group}. The default values in ./params/resnet.txt correspond to ResNet18.<br/>
`./params/vit.txt`: Vision Transformer architectures can be specified here. The order of arguments is: {[patch_size, num_layers, num_heads], [hidden_dim, mlp_dim], [dropout, attention_dropout], image_size}.<br/>
`./params/wideresnet.txt`: WideResNet architectures can be specified here. The order of arguments is: {depth, widen_factor, dropout}. The default values in ./params/wideresnet.txt correspond to WideResNet28-2.<br/>
`./params/fc.txt`: Fully Connected network architectures can be specified here. The order of arguments is: {num_units, num_units, ...}, so that each entry corresponds to the number of units for each successive layer. The default values in ./params/fc.txt correspond to a fully connected network with two layers, each with 784 units.<br/>

## Usage:
As a very basic example of usage, to train and validate a fully connected network with normality normalization on the MNIST classification task, enter:

```python train.py```

See `config.py` for the comprehensive list of arguments which can be passed in, including passing the neural network architecture as an argument (by default it is set to a ReLU activated neural network with 2 hidden layers).

By default, all outputs are saved to folder `./runs/`, and grouped by the model name of the run (which by default is the date and time the run was launched). Inspecting folder `./runs/` during/after a run, should make clear what is saved by default during the run. If you'd simply like to inspect the performance of the model during/after the run, opening the stdout file in the corresponding run's `stdout` folder will suffice.

To reproduce the ResNet18/CIFAR10 experimental setup, enter:

```python train.py --base_config_path=./configs/model_default_configs/resnet.yaml --dataset=CIFAR10```

To reproduce the ViT/SVHN experimental setup, enter:

```python train.py --base_config_path=./configs/model_default_configs/vit.yaml --dataset=SVHN```

In a similar manner, our experimental configurations can be reproduced using these simple command line changes, and for architectural changes, by modifying the corresponding `./params/<model_type>.txt` file.

You can also simply copy our `./layers.py` file into your own codebase, and import any of the layers we've implemented from there. For example, to import `BatchNormalNorm2d`, you would enter `from layers import BatchNormalNorm2d`.

### License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
