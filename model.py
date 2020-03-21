from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch
import torch.nn as nn


def get_model(config):

    criterion = nn.CrossEntropyLoss()

    if config['pretrained'] is True:

      if 'densenet121' in config['model']:
        model = models.densenet121(pretrained=True)

      if 'densenet201' in config['model']:
        model = models.densenet201(pretrained=True)

      elif 'efficientnetb0' in config['model']:
        model = EfficientNet.from_pretrained('efficientnet-b0')

      elif 'efficientnetb1' in config['model']:
        model = EfficientNet.from_pretrained('efficientnet-b1')

      elif 'shufflenet' in config['model']:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=True)

    else:
      if 'densenet121'in config['model']:
        model = models.densenet121(pretrained=False)

      if 'densenet201' in config['model']:
        model = models.densenet201(pretrained=False)

      elif 'efficientnetb0' in config['model']:
        model = EfficientNet.from_name('efficientnet-b0')

      elif 'efficientnetb1' in config['model']:
        model = EfficientNet.from_name('efficientnet-b1')

      elif 'shufflenet' in config['model']:
        model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=False)

    if config['added_layers'] is 'dropout' and config['model'].lower() is 'densenet121':
      num_ftrs = model.classifier.in_features
      model.classifier = nn.Sequential(
          nn.Dropout(0.5),
          nn.Linear(num_ftrs, 1))

    if config['added_layers'] is 'linear' and config['model'].lower() is 'densenet121':
      num_ftrs = model.classifier.in_features
      model.classifier = nn.Sequential(
          nn.Linear(num_ftrs, 512),
          nn.Linear(512, 256),
          nn.Linear(256, 5))

    return model, criterion
