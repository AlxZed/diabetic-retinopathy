config = {
    'model': 'densenet121',
    'pretrained': True,
    'feature_extraction': False,
    'balanced_sampler': False,
    'batch_size': 16,
    'optimizer': 'adam',
    'lr': 0.0004,
    'scheduler': 'custom',
    'weight_decay': '',
    'added_layers': '',
    'transformations': 5,
    'random_crop_size': 128,
    'pooling': 'mean'
    'n_epochs': 50,
}