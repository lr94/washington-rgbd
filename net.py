import torch.nn as nn
import torchvision.models


def init_network(n_classes, pretrained=False):
    resnet = torchvision.models.resnet18(pretrained=pretrained)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
    return resnet
