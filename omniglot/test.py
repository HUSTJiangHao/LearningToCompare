# 功能测试文件，和功能无关

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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