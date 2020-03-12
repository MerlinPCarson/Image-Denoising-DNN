import torch.nn as nn
import torch
import math


def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
    if isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(layer.bias.data, 0.0)

class CNN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(CNN_ReLU, self).__init__()
        padding = int((filter_size - 1) / 2)
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, filter_size, padding=padding, bias=False), nn.ReLU())
		
    def forward(self, x):
        return self.layer(x)


class CNN_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(CNN_BN_ReLU, self).__init__()
        padding = int((filter_size - 1) / 2)
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, filter_size, padding=padding, bias=False),\
		                           nn.BatchNorm2d(in_channels),\
		                           nn.ReLU())
		
    def forward(self, x):
        return self.layer(x)
	
class CNN_BN_ReLU_Res(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(CNN_BN_ReLU_Res, self).__init__()
        padding = int((filter_size - 1) / 2)
        self.layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, filter_size, padding=padding, bias=False),\
		                           nn.BatchNorm2d(in_channels),\
		                           nn.ReLU(),\
                                   nn.Conv2d(in_channels, out_channels, filter_size, padding=padding, bias=False),\
		                           nn.BatchNorm2d(in_channels))
		
    def forward(self, x):
        return nn.functional.relu(x + self.layer(x))
	

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(CNN, self).__init__()
        padding = int((filter_size - 1) / 2)
        self.layer = nn.Conv2d(in_channels, out_channels, filter_size, padding=padding, bias=False)
		
    def forward(self, x):
        return self.layer(x)

class DnCNN(nn.Module):
    def __init__(self, num_layers, input_channels, output_channels, filter_size):
        super(DnCNN, self).__init__()
        self.layers = nn.Sequential(
        CNN_ReLU(input_channels, output_channels, filter_size),
        nn.Sequential(*[CNN_BN_ReLU(output_channels, output_channels, filter_size) for x in range(num_layers)]),
        CNN(output_channels, input_channels, filter_size))
		
    def forward(self, x):
        return self.layers(x)	

class DnCNN_Res(nn.Module):
    def __init__(self, num_layers, input_channels, output_channels, filter_size):
        super(DnCNN_Res, self).__init__()
        self.layers = nn.Sequential(
        CNN_ReLU(input_channels, output_channels, filter_size),
        nn.Sequential(*[CNN_BN_ReLU_Res(output_channels, output_channels, filter_size) for x in range(num_layers)]),
        CNN(output_channels, input_channels, filter_size))
		
    def forward(self, x):
        return self.layers(x)	
