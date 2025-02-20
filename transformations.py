from torchvision import transforms
from torchvision import datasets


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path


def get_transformations(config):

    _mean = [0.4432, 0.3067, 0.2193]
    _std = [0.203, 0.1411, 0.1004]

    if config['transformations'] == 5:
        train_trans = transforms.Compose([
          transforms.RandomCrop(config['random_crop_size']),
          transforms.RandomHorizontalFlip(),
          transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
          transforms.RandomAffine(degrees=(-180, 180),scale=(0.8889, 1.0),shear=(-36, 36)),
          transforms.ColorJitter(contrast=(0.9, 1.1)),
          transforms.ToTensor(),
          transforms.Normalize(_mean, _std)])
    
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ])

    elif config['transformations'] == 3:
        train_trans = transforms.Compose([
            transforms.RandomCrop(config['random_crop_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std)])
    
        val_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_mean, _std),
        ])
        
    elif config['transformations'] == 1:
        train_trans = transforms.Compose([
            transforms.RandomCrop(config['random_crop_size']),
            transforms.ToTensor(),
    
        val_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    return train_trans, val_trans


def get_datasets(config, dataset_path, train_trans, val_trans):

    if config['reduce_class_0']:
        train_ds = ImageFolderWithPaths(f"{dataset_path}/reduced/train/", transform=train_trans)
        val_ds = ImageFolderWithPaths(f"{dataset_path}/reduced/val/", transform=val_trans)
        test_ds = ImageFolderWithPaths(f"{dataset_path}/reduced/test/", transform=val_trans)

    else:
        train_ds = ImageFolderWithPaths(f"{dataset_path}/full/train/", transform=train_trans)
        val_ds = ImageFolderWithPaths(f"{dataset_path}/full/val/", transform=val_trans)
        test_ds = ImageFolderWithPaths(f"{dataset_path}/full/test/", transform=val_trans)

    print(f'Total Train Images: {len(train_ds)}')
    print(f'Total Val Images: {len(val_ds)}')
    print(f'Total Test Images: {len(test_ds)}')

    return train_ds, val_ds, test_ds
