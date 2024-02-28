from torch.optim import SGD, RMSprop
from torchvision import models
from torch import nn
import torch

def loss():
    return nn.CrossEntropyLoss()

def optimizer(params=None,lr:float=0.01):
    return RMSprop(params=params, lr=lr)

def device(device:str = None) -> str:
    if device==None or '':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        return device

epochs = 10
UPLOAD_DIR = "static"