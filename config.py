config = {
    'model': 'densenet121',
    'pretrained': True,
    'batch_size': 32,
    'optimizer': 'adam',
    'lr': 0.01,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'loaded_weigths': '',
    'weight_decay': '',
    'added_layers': '',
    'img_size': 512,
    'balanced': True,
    'transformations': 5,
}