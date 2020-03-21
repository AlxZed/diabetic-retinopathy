from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler


def get_dataloaders(config, train_ds, val_ds, test_ds):

    if config['balanced'] is True:

        weighted_sampler = True

        if weighted_sampler:
            sampler = ImbalancedDatasetSampler(train_ds)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dl = DataLoader(
                train_ds,
                batch_size=config['batch_size'],
                num_workers=12,
                pin_memory=True,
                sampler=sampler,
                shuffle=shuffle
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

    return train_dl, val_dl, test_dl