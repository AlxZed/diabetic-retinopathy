import torch


def get_weights(config, model):

    if config['weight_loading'] != '':
      model.load_state_dict(torch.load(config['weight_loading']))
