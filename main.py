import torch
import os
import pytz
import wandb

from transformations import get_transformations, get_datasets
from data_loaders import get_dataloaders
from model import get_model, get_loss, get_added_layers, get_pooling, get_features
from train import training_loop
from optimizer_scheduler import get_optimizer, get_scheduler
from weights import get_state_dict
from config import config
from datetime import datetime


def get_experiment_prefix(config):
    tz_NY = pytz.timezone('America/New_York')
    datetime_NY = datetime.now(tz_NY)
    prefix = f"{config['model']}_{datetime_NY.strftime('%D')}"
    return prefix


def main():
    train_trans, val_trans = get_transformations(config)
    train_ds, val_ds, test_ds = get_datasets(config, dataset_path, train_trans, val_trans)
    train_dl, val_dl, test_dl = get_dataloaders(config, train_ds, val_ds, test_ds)
    model = get_model(config)
    model = get_added_layers(config, model)
    get_pooling(config)
    criterion = get_loss(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    model = model.to(DEVICE)
    get_state_dict(config, model)
    get_features(config, model)

    prefix = get_experiment_prefix(config)

    wandb.init(project="april_2020", name=prefix, config=config)

    training_loop(model, optimizer, scheduler, val_dl, train_dl, criterion, config, DEVICE, prefix, timezone)


if __name__ == '__main__':

    timezone = pytz.timezone('America/New_York')

    # logging wandb session
    os.system("wandb login 91b90542bdde4812440c8b554b")

    # initialize CUDA
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    # check if balanced sampler was installed
    if os.path.isdir('./imbalanced-dataset-sampler') is False:
        os.system('git clone https://github.com/ufoym/imbalanced-dataset-sampler.git')
        os.system('cd imbalanced-dataset-sampler && python setup.py install && pip install .')

    # dataset path
    dataset_path = './Desktop/'

    main()
