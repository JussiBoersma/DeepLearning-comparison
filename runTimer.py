import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import time
import csv
import warnings
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

start = timer()
from models.SimpleNN import simplenn
end = timer()
print('simplenn:')
print(end - start)

start = timer()
from models.AlexNet import pipe_alexnet
end = timer()
print('alexnet:')
print(end - start)

start = timer()
from models.GoogleNet import googlenet
end = timer()
print('googlenet:')
print(end - start)

start = timer()
from models.ResNet import resnet
end = timer()
print('resnet:')
print(end - start)

start = timer()
from models.ShuffleNet import shufflenet
end = timer()
print('shuffle:')
print(end - start)

start = timer()
from models.VGG import vgg
end = timer()
print('vgg:')
print(end - start)











