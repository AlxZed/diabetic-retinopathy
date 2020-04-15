import torch.nn as nn


def get_loss(config):
    if 'cross' in config['loss'].lower():
        criterion = nn.CrossEntropyLoss()

    elif 'mse' in config['loss'].lower():
        criterion = nn.MSELoss()

    else:
        print('loss not specified')

    return criterion

