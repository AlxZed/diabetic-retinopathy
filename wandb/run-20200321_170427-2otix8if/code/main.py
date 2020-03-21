from transformations import get_transformations
from data_loaders import get_dataloaders
from model import get_model
from train import training_loop
from optimizer_scheduler import get_optimizer, get_scheduler
from weights import get_weights
from config import config
import datetime
import torch
import os
import pytz
import wandb


def intitialize_wandb_session(datetime):
    date_object = datetime.date.today()

    from datetime import datetime
    tz_NY = pytz.timezone('America/New_York')
    datetime_NY = datetime.now(tz_NY)
    prefix = f"{date_object}_{datetime_NY.strftime('%H:%M:%S')}"
    config = wandb.config
    wandb.init(project="march_2020", name=prefix, config=config)

    return prefix


def main():
    train_ds, val_ds, test_ds = get_transformations(config, dataset_path)
    train_dl, val_dl, test_dl = get_dataloaders(config, train_ds, val_ds, test_ds)
    model, criterion = get_model(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    model = model.to(DEVICE)
    get_weights(config, model)
    training_loop(model, optimizer, scheduler, val_dl, train_dl, criterion, config, DEVICE, prefix, tz_NY)


if __name__ == '__main__':

    # logging wandb session
    os.system("wandb login 91b90542bdde4812440c8b554b")

    # # initialize CUDA
    # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.get_device_name(0))

    #dataset path
    dataset_path = './Desktop/'

    prefix = intitialize_wandb_session(datetime)

    main()
