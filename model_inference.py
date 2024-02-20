import torch
from steps import test_step
from data_transformation import image_transform

model = torch.load(f = 'vgg16.pt')
input = image_transform
model(input)