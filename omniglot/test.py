from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("1.png")

image = image.convert('L')
image = image.resize((28,28), resample=Image.LANCZOS)
