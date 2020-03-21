import torch


def get_weights(config, model):

    if config['loaded_weigths'] is not '':
      model.load_state_dict(torch.load('./past_weights.pt'))
