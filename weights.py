import torch


def get_state_dict(config, model):

    if '.pt' in config['state_dict_path']:
      model.load_state_dict(torch.load(config['state_dict_path']))
