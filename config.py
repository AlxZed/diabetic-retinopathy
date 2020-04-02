config = {
    'model': 'densenet121',
    'pretrained': True,
    'feature_extraction': False,
    'balanced_sampler': False,
    'batch_size': 16,
    'optimizer': 'adam',
    'lr': 0.0002,
    'scheduler': 'custom',
    'state_dict_path': '',
    'added_layers': '',
    'transformations': 5,
    'random_crop_size': 256,
    'pooling': 'mean',
    'n_epochs': 100,
    'reduce_class_0': True,
}
