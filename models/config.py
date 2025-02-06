import importlib

dataset_params = {
    'mnist': 10,
    'cifar10': 10,
    'food101': 101,
}

model_params = {
    'mnist': {
        'mlp': {'dim_in': 784,
                'hidden_layer': 256},
        'cnnmnist': {}
    },
    'cifar10': {
        'cnncifar': {}
    },
    'food101': {
        'mlaresnet18': {'num_classes': 101,
                        'vocab_size': 30052,
                        'embedding_dim': 256},
        'mlaresnet34': {'num_classes': 101,
                        'vocab_size': 30052,
                        'embedding_dim': 256},
        'mlaresnet50': {'num_classes': 101,
                        'vocab_size': 30052,
                        'embedding_dim': 256},
        'mmresnet18': {'num_classes': 101,
                       'vocab_size': 30052,
                       'embedding_dim': 256},
    }
}


def load_model(args):
    model_arg = args.model
    dataset_arg = args.dataset.split('-')[0]
    args.class_num = dataset_params[dataset_arg]

    if dataset_arg not in model_params.keys():
        exit('Dataset params not exist (in config.py)!')

    params = None
    if model_arg in model_params[dataset_arg].keys():
        params = {**model_params[dataset_arg][model_arg]}

    model_module = importlib.import_module(f'models.{model_arg}')
    return getattr(model_module, model_arg)(args, params)
