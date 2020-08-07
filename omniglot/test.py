# 功能测试文件，和功能无关

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
"""
image = Image.open("1.png")

image = image.convert('L')
image = image.resize((28,28), resample=Image.LANCZOS)
rotation = 60

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

rotate = Rotate(rotation)

image = rotate(image)
image.show()

"""
"""
loss_fn = torch.nn.MSELoss()
input = torch.autograd.Variable(torch.randn(3,4))
target = torch.autograd.Variable(torch.randn(3,4))
loss = loss_fn(input, target)
print(input); print(target); print(loss)
print(input.size(), target.size(), loss.size())
"""


t_out = torch.randn(1,500,500)
img1 = transforms.ToPILImage()(t_out)
img1.show()