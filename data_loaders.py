from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler


def get_dataloaders(config, train_ds, val_ds, test_ds):

    if config['balanced_sampler'] is True:

        train_dl = DataLoader(
                train_ds,
                batch_size=config['batch_size'],
                num_workers=12,
                pin_memory=True,
                sampler=ImbalancedDatasetSampler(train_ds),
                shuffle=False
        )

        val_dl = DataLoader(
                val_ds,
                batch_size=config['batch_size'],
                num_workers=12,
                shuffle=False,
                pin_memory=True
        )

        test_dl = DataLoader(
                test_ds,
                batch_size=config['batch_size'],
                num_workers=12,
                shuffle=False,
                pin_memory=True
        )

    else:
        train_dl = DataLoader(train_ds,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=12)

        val_dl = DataLoader(val_ds,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=12)

    return train_dl, val_dl, test_dl