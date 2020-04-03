from efficientnet_pytorch import EfficientNet
from torchvision import models
from torch.nn.parameter import Parameter
import torch
import os
import torch.nn as nn


def get_model(config):

    if config['pretrained'] is True:

        if 'resnet18' in config['model']:
            model = models.resnet18(pretrained=True)

        elif 'densenet121' in config['model']:
            model = models.densenet121(pretrained=True)

        elif 'densenet201' in config['model']:
            model = models.densenet201(pretrained=True)

        elif 'efficientnetb0' in config['model']:
            model = EfficientNet.from_pretrained('efficientnet-b0')

        elif 'efficientnetb1' in config['model']:
            model = EfficientNet.from_pretrained('efficientnet-b1')

        elif 'shufflenet' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=True)

    else:
        if 'densenet121' in config['model']:
            model = models.densenet121(pretrained=False)

        elif 'densenet201' in config['model']:
            model = models.densenet201(pretrained=False)

        elif 'efficientnetb0' in config['model']:
            model = EfficientNet.from_name('efficientnet-b0')

        elif 'efficientnetb1' in config['model']:
            model = EfficientNet.from_name('efficientnet-b1')

        elif 'efficientnetb2' in config['model']:
            model = EfficientNet.from_name('efficientnet-b2')
            
        elif 'efficientnetb3' in config['model']:
            model = EfficientNet.from_name('efficientnet-b3')

        elif 'shufflenet' in config['model']:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=False)
    return model


def get_loss(config):
    criterion = nn.CrossEntropyLoss()
    return criterion


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


def get_pooling(config):

    if 'mean' in config['pooling']:
        def gem(x, p=3, eps=1e-6):
            return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

        class GeM(nn.Module):
            def __init__(self, p=3, eps=1e-6):
                super(GeM, self).__init__()
                self.p = Parameter(torch.ones(1) * p)
                self.eps = eps

            def forward(self, x):
                return gem(x, p=self.p, eps=self.eps)

            def __repr__(self):
                return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
                        self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
