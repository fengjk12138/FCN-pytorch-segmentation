import matplotlib.pyplot as plt
import numpy
import torch
import torchvision

tmp = torch.load("./PNG/out9.pkl")
x = tmp[0]
y = tmp[1]

for i in range(x.size()[0]):
    plt.subplot(121)
    plt.imshow(y[i])
    plt.subplot(122)
    plt.imshow(x[i])
    print(y[i].max(), x[i].max())
    plt.waitforbuttonpress()
    plt.close()