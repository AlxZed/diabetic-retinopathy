import torch
from torch.optim import lr_scheduler


def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def get_optimizer(config, model):
    if config['optimizer'] is 'adam':
        optimizer = torch.optim.Adam(get_trainable(model.parameters()), lr=config['lr'])
        return optimizer


def get_scheduler(config, optimizer):
    if config['scheduler'] is 'StepLR':
        step_size = 5
        gamma = 0.464159

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        config['step_size'] = step_size
        config['gamma'] = gamma

    elif config['scheduler'] is 'ReduceLROnPlateau':
        mode = 'max'
        factor = 0.25
        patience = 3

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
        config['mode'] = mode
        config['factor'] = factor
        config['patience'] = patience

    elif config['scheduler'] is 'CosineAnnealingWarmRestarts':
        T_0 = 10
        T_mult = 1
        eta_min = 0
        last_epoch = -1

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
                                                             last_epoch=last_epoch)
        config['T_0'] = T_0
        config['T_mult'] = T_mult
        config['eta_min'] = eta_min
        config['last_epoch'] = last_epoch

    return scheduler