from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch
import os
import torch.nn as nn


def get_model(config):

    if config['pretrained'] is True:

        if 'resnet18' in config['model']:
            model = models.resnet18(pretrained=True)

        if 'resnet50' in config['model']:
            model = models.resnet50(pretrained=True)

        elif '121' in config['model']:
            model = models.densenet121(pretrained=True)

        elif '201' in config['model']:
            model = models.densenet201(pretrained=True)

        elif 'b0' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b0')

        elif 'b1' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b1')

        elif 'b2' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b2')

        elif 'b3' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b3')

        elif 'b4' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b4')

        elif 'b5' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b5')

        elif 'b6' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b6')

        elif 'b7' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_pretrained('efficientnet-b7')

        elif 'shuffle' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=True)

        elif 'resnext50_32x4d' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext50_32x4d', pretrained=True)

    else:

        if 'resnet18' in config['model']:
            model = models.resnet18(pretrained=False)

        if 'resnet50' in config['model']:
            model = models.resnet50(pretrained=False)

        elif '121' in config['model']:
            model = models.densenet121(pretrained=False)

        elif '201' in config['model']:
            model = models.densenet201(pretrained=False)

        elif 'b0' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b0')

        elif 'b1' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b1')

        elif 'b2' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b2')

        elif 'b3' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b3')

        elif 'b4' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b4')

        elif 'b5' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b5')

        elif 'b6' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b6')

        elif 'b7' in config['model']:
            os.system('pip install efficientnet_pytorch')
            model = EfficientNet.from_name('efficientnet-b7')

        elif 'shufflenet' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=False)

        elif 'resnext50_32x4d' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext50_32x4d', pretrained=False)

    return model


def get_features(config, model):
    if config['feature_extraction'] is True:
        # Freeze all parameters:
        for param in model.parameters():
            param.requires_grad = False


def get_added_layers(config, model):
    if 'dropout' in config['added_layers'] and 'densenet121' in config['model'].lower():
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1))

    if 'linear' in config['added_layers'] and 'densenet121' in config['model'].lower():
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
          nn.Linear(num_ftrs, 512),
          nn.Linear(512, 256),
          nn.Linear(256, 5))

    return model
