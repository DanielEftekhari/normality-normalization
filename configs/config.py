import argparse


class StoreDictKeyPair(argparse.Action):
     def __call__(self, parser, namespace, values, option_string=None):
         my_dict = {}
         for kv in values.split(','):
             k,v = kv.split('=')
             my_dict[k] = v
         setattr(namespace, self.dest, my_dict)


def checkbool(x):
    return x.lower() != 'false'


def training():
    # initialize parser
    parser = argparse.ArgumentParser()

    # config params
    config_arg = parser.add_argument_group('Config Params')
    config_arg.add_argument('--base_config_path', type=str, default='./configs/default_config.yaml',
                            help='yaml configuration file, to initialize configs with')
    config_arg.add_argument('--custom_config_path', type=str, default=None,
                            help='yaml configuration file with any user-specified configs, to overwrite the base config values with;'
                            'if this is a valid yaml filepath, the parameters therein will override their respective command-line config arguments')

    # general model params
    model_arg = parser.add_argument_group('Model Params')
    model_arg.add_argument('--model_type', type=str,
                           help='whether to use a:'
                           'fully connected network (MLP): <fc>'
                           'a residual network (ResNet): <resnet>'
                           'a vision transformer (ViT): <vit>'
                           'a wide residual network (WideResNet): <wideresnet>'
                           )
    model_arg.add_argument('--param', nargs='+',
                           help=
                           'if --param is set, it overrides the parameters read from the --param_dir argument directory')
    model_arg.add_argument('--param_dir', type=str,
                           help='path of directory to txt file containing default model architecture specifications;'
                           'FC: the filename should be fc_param.txt'
                           'ResNet: the filename should be resnet_param.txt'
                           'WideResNet: the filename should be    wideresnet_param.txt'
                           'ViT: the filename should be vit_param.txt'
                           'default examples are provided in the ./params/ folder'
                           'this option is overruled if --param is set')
    model_arg.add_argument('--activation', type=str,
                           help='activation function; one of <sigmoid>, <tanh>, <relu>, <linear>, <gelu>')
    model_arg.add_argument('--norm', type=str,
                           help='normalization; one of <normal> (BatchNormalNorm AND LayerNormalNorm), <batch> (BatchNorm), <layer> (LayerNorm), <none> (no norm)')

    # dataset params
    dataset_arg = parser.add_argument_group('Data Params')
    dataset_arg.add_argument('--data_dir', type=str,
                             help='directory to load/save dataset from/to')
    dataset_arg.add_argument('--dataset', type=str,
                             help='dataset to use')

    # criterion params
    criterion_arg = parser.add_argument_group('Criterion Params')
    criterion_arg.add_argument('--criterion', type=str,
                               help='loss/criterion function, one of'
                               '<ce> cross-entropy loss (classification)'
                               '<l2> mean-squared error (regression)'
                               '<l1> absolute error (regression)')

    # training params
    train_arg = parser.add_argument_group('Training Params')
    train_arg.add_argument('--order', type=int,
                            help='when estimating lambda, whether to use a first(1) or second(2) order optimization algorithm')
    train_arg.add_argument('--alpha', type=float,
                           help='learning rate for estimating lambda in normality normalization')
    train_arg.add_argument('--iterations', type=int,
                           help='number of update steps to use, when estimating lambda in normality normalization')
    train_arg.add_argument('--eps', type=float,
                           help='value of epsilon to add to divisors, to prevent numerical instability')
    train_arg.add_argument('--noise_train', type=float,
                            help='standard deviation of a normal distribution, representing the amount of perturbative noise to add to each unit\'s pre-activation, in each layer; at train time')
    train_arg.add_argument('--weighted_sampler', type=checkbool,
                            help='whether to use a weighted sampler for handling class imbalance')
    train_arg.add_argument('--dropout', type=float,
                           help='dropout rate to apply during training'
                           '<0> for no dropout')
    train_arg.add_argument('--epochs', type=int,
                            help='number of epochs to train for')
    train_arg.add_argument('--batch_size', type=int,
                            help='minibatch size during training and evaluation')
    train_arg.add_argument('--optim', type=str,
                           help='optimizer; one of <sgd>, <adam>, <adamw>')
    train_arg.add_argument('--lr', type=float,
                           help='learning rate')
    train_arg.add_argument('--weight_decay', type=float,
                           help='weight decay (l2-regularization) term')
    train_arg.add_argument('--momentum', type=float,
                           help='momentum for sgd optimizer')
    train_arg.add_argument('--nesterov', type=checkbool,
                            help='whether to use nesterov momentum with sgd')
    train_arg.add_argument('--beta1', type=float,
                           help='beta1 for adam/adamw optimizer')
    train_arg.add_argument('--beta2', type=float,
                           help='beta2 for adam/adamw optimizer')
    train_arg.add_argument('--shuffle', type=checkbool,
                           help='whether to shuffle training data')
    train_arg.add_argument('--adv_train', type=checkbool,
                           help='whether to train with regular training and adversarial training')
    train_arg.add_argument('--adv_eps', type=float,
                           help='adversarial training epsilon')
    train_arg.add_argument('--adv_eps_iter', type=float,
                           help='adversarial training step size for each attack iteration')
    train_arg.add_argument('--adv_nb_iter', type=int,
                           help='adversarial training number of attack iterations')
    train_arg.add_argument('--adv_norm', type=str,
                           help='adversarial training norm; one of <1>, <2>, <inf>')

    # evaluation params
    eval_arg = parser.add_argument_group('Evaluation Params')
    eval_arg.add_argument('--adv_eval', type=checkbool,
                           help='whether to evaluate on adversarial examples')

    # runtime params
    runtime_arg = parser.add_argument_group('Runtime')
    runtime_arg.add_argument('--num_workers', type=int,
                             help='number of workers/subprocesses to use in dataloader')
    runtime_arg.add_argument('--device', type=str,
                             help='device; one of <cuda>, <cpu>')
    runtime_arg.add_argument('--random_seed', type=int,
                             help='seed for reproducibility')
    runtime_arg.add_argument('--debug_mode', type=checkbool,
                             help='enable pytorch debug mode')

    # logging params
    logging_arg = parser.add_argument_group('Logging')
    logging_arg.add_argument('--model_name', type=str,
                             help='model name')
    logging_arg.add_argument('--log_dir', type=str,
                             help='root directory for logging')
    logging_arg.add_argument('--tmp_log_dir', type=str,
                             help='root directory for temporary logging')
    logging_arg.add_argument('--stdout_dir', type=str,
                             help='directory to log program stdout to')
    logging_arg.add_argument('--config_dir', type=str,
                             help='directory to log program config to')
    logging_arg.add_argument('--metric_dir', type=str,
                             help='directory to log performance metrics to')
    logging_arg.add_argument('--save_model', type=str,
                             help='scheme for saving model weights; one of:'
                             '<best> (save the running best model in terms of validation performance, and latest model)'
                             '<all> (save the model weights at the end of each epoch),'
                             '<none> (model weights not saved)')
    logging_arg.add_argument('--model_dir', type=str,
                             help='directory in which to save model checkpoints')
    logging_arg.add_argument('--plot', type=checkbool,
                             help='whether to plot basic performance metrics')
    logging_arg.add_argument('--plot_dir', type=str,
                             help='directory in which to save plots')

    return parser
