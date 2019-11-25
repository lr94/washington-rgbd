import torch
import torch.nn as nn
import torchvision.models


def init_network(n_classes, pretrained=False, input_channels=3):
    resnet: torchvision.models.ResNet = torchvision.models.resnet18(pretrained=pretrained)
    resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)

    # Change the number of channels
    if input_channels != resnet.conv1.in_channels:
        assert (input_channels == 6 or not pretrained) and resnet.conv1.in_channels == 3
        # Create new convolutional layer
        new_conv1 = nn.Conv2d(input_channels, resnet.conv1.out_channels, kernel_size=resnet.conv1.kernel_size,
                              stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)

        # Duplicate the pretrained weights
        if pretrained:
            new_weights = torch.cat((resnet.conv1.weight, resnet.conv1.weight), dim=1)
            new_conv1.weight = nn.Parameter(new_weights, requires_grad=resnet.conv1.weight.requires_grad)

        resnet.conv1 = new_conv1

    return resnet
