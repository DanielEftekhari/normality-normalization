import sys
import os
import itertools
import collections

sys.path.insert(0, '..')
import utils


def get_membership(d, parent=None, membership={}):
    """
    recursively maps the child-to-parent mappings in a dict object
    """
    for key in d:
        membership[key] = parent
        if type(d[key]) == dict:
            get_membership(d[key], key, membership)
    return membership


def update_params(params, args, membership):
    """
    updates <params> with the key-value pairs in <args>
    """
    for key in args:
        # check that the key is a relevant parameter
        if key not in membership:
            continue

        # trace the key back to the root of the object
        tree = [key]
        while key not in params:
            key = membership[key]
            tree.append(key)

        # descend the tree, then update the parameter's value
        d = params
        for i in range(len(tree)-1, 0, -1):
            d = d[tree[i]]
        d[tree[0]] = args[tree[0]]

    return params


def merge(mapping, renumber=True):
    # keep track of duplicate runs
    table = collections.defaultdict(list)
    for id_ in mapping:
        cfg = mapping[id_]
        table[str(cfg)].append(id_)

    # delete duplicate runs
    for key in table:
        if len(table[key]) > 1:
            for j in range(1, len(table[key])):
                del mapping[table[key][j]]

    if renumber: # re-number run ids
        mapping_renumbered = {}
        counter = 0
        for run_id in mapping:
            mapping_renumbered[counter+1] = mapping[run_id]
            counter += 1
        return mapping_renumbered
    return mapping


def prune(mapping, filters):
    for id_ in mapping:
        for filter in filters: # ex: filter='norm'
            if filter in mapping[id_]:
                for arg in filters[filter]: # ex: arg='normal'
                    if mapping[id_][filter] == arg:
                        for val in filters[filter][arg]:
                            if val in mapping[id_]:
                                del mapping[id_][val]
    return mapping


def product(d):
    return list(itertools.product(*list(d.values())))


def product_with_map(d):
    args = list(d.keys())
    vals = product(d)
    num_runs = len(vals)

    res = {}
    for id_ in range(num_runs):
        cfg = vals[id_]
        res[id_+1] = {args[j]: cfg[j] for j in range(len(args))}
    return res


def override_options(d, options):
    if options is not None:
        for option in options:
            if options[option].isnumeric():
                elem = int(options[option])
            elif utils.can_be_float(options[option]):
                elem = float(options[option])
            else:
                elem = options[option]
            d[option] = [elem]
    return d


def make_config_rn18_cifar10(options={}):
    d, filters= {}, {}

    d['model_type'] = ['resnet',]
    d['param'] = [
                  [['BasicBlock'], [2, 2, 2, 2], [64]]
                 ]

    d['activation'] = ['relu',]
    d['norm'] = ['normal', 'batch']

    d['dataset'] = ['CIFAR10']

    d['order'] = [2]
    d['alpha'] = [1.0]
    d['iterations'] = [1]
    d['noise_train'] = [0.4]
    d['eps'] = [1e-05]

    d['epochs'] = [200]
    d['batch_size'] = [128]
    d['optim'] = ['sgd']
    d['lr'] = [0.1]
    d['weight_decay'] = [5e-04]
    d['momentum'] = [0.9]
    d['nesterov'] = [False]

    d['criterion'] = ['ce']

    d['random_seed'] = [2]
    # d['random_seed'] = [2, 3, 5, 7, 11, 13] # for running more experiments with differing random seeds

    d['save_model'] = ['best']
    d['plot'] = [True]

    filters['norm'] = {'none': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],
                       'batch': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],
                       'layer': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],}

    # override defaults
    d = override_options(d, options)

    return d, filters


def make_config_vit_svhn(options={}):
    d, filters= {}, {}

    d['model_type'] = ['vit']
    d['param'] = [
                  [
                    [4, 8, 8], # [patch_size, num_layers, num_heads]
                    [768, 2304], # [hidden_dim, mlp_dim]
                    [0.0, 0.0], # [dropout, attention_dropout]
                    [32], # [image_size]
                   ],
                 ]

    d['norm'] = ['normal', 'layer']

    d['dataset'] = ['SVHN']

    d['order'] = [2]
    d['alpha'] = [1.0]
    d['iterations'] = [1]
    d['noise_train'] = [1.0]
    d['eps'] = [1e-05]

    d['epochs'] = [200]
    d['batch_size'] = [32]
    d['optim'] = ['adamw']
    d['lr'] = [1e-03]

    d['criterion'] = ['ce']

    d['random_seed'] = [2]
    # d['random_seed'] = [2, 3, 5, 7, 11, 13] # for running more experiments with differing random seeds

    d['save_model'] = ['best']
    d['plot'] = [True]

    filters['norm'] = {'none': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],
                       'batch': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],
                       'layer': ['alpha', 'iterations', 'order', 'eps', 'noise_train'],}

    # override defaults
    d = override_options(d, options)

    return d, filters


if __name__ == '__main__':
    import argparse
    from config import StoreDictKeyPair

    utils.make_dirs('./stdout/', replace=False)
    sys.stdout = open('./stdout/stdout_{}.txt'.format(utils.get_current_time()), 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--function_name', type=str, required=True)
    parser.add_argument('--options', dest='options', action=StoreDictKeyPair, required=False)
    config, unparsed = parser.parse_known_args()

    runs_dir = './runs/{}/'.format(config.run_name)
    utils.make_dirs(runs_dir, replace=True)

    func = globals()['make_config_{}'.format(config.function_name)]

    d, filters = func(config.options)
    mapping = product_with_map(d)
    utils.save_yaml(mapping, os.path.join(runs_dir, 'mapping.yaml'))

    pruned_mapping = prune(mapping, filters)
    filtered_mapping = merge(pruned_mapping)
    utils.save_yaml(filtered_mapping, os.path.join(runs_dir, 'runs.yaml'))

    for id_ in filtered_mapping:
        utils.save_yaml(filtered_mapping[id_], os.path.join(runs_dir, 'cfg_{}.yaml'.format(id_)))

    run_ids = list(filtered_mapping.keys())
    run_ids = [[str(id_)] for id_ in run_ids]
    utils.save_txt(run_ids, os.path.join(runs_dir, 'table.dat'))
