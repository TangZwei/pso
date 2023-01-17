import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from matplotlib import pyplot as plt
from PSO import *


def prediction(net, lam, radius):
    X = torch.dstack((lam, radius)).reshape((lam.shape[0], 2))
    y_hat = net(X)
    return y_hat


def eval_acc(y_hat, y, threshold=0.05):
    acc = torch.count_nonzero(
        (torch.abs(y_hat.reshape(y.shape) - y) / y) < threshold)
    return acc / y.numel()


net = torch.load("D:\\net_lr_decay_(30_0.9).pth")

fn = 'D:\\Researches\\FDTD_simulations\\Training_data\\Si_pillar\\data\\Si_pillar_radius_test1.txt'
raw = pd.read_table(fn, sep='\t', header=None)
# wavelength, Transmission
test_set_1 = torch.asarray(raw.values, dtype=torch.float32)
wavelength = test_set_1[:, 0]
radius = 0.21 * torch.ones_like(wavelength)
transmission1 = prediction(net, wavelength, radius)
with torch.no_grad():
    acc1 = eval_acc(transmission1, test_set_1[:, 1])
print(acc1)
ax = plt.subplot(1, 2, 1)
ax.plot(wavelength, transmission1.detach().numpy(),
        wavelength, test_set_1[:, 1])
# plt.show()

def single_loss(x, target=0.613882):
    X = torch.asarray(x, dtype=torch.float32)
    with torch.no_grad():
        pred = net(X)
    return torch.abs(pred - target).numpy()

# ls = loss([0.77, 0.23], 1)

# find a 2d-point (wavelength, radius) with a desired transmission
pso = PSO(dim=2, size=1000, num_iter=1000, min_pos=[0.75, 0.05], max_pos=[0.85, 0.25],
          min_vel=[-0.05, -0.1], max_vel=[0.05, 0.1], tolerance=1e-3, loss=single_loss, C1=2, C2=2, W=1)
fitnessList, positionList, bestPos = pso.update_ndim()
print('wavelength: %.8f, radius: %.8f' % (bestPos[0], bestPos[1]))

# find a 1d-point (radius) with a desired spectrum
fn = 'D:\\Researches\\FDTD_simulations\\Training_data\\Si_pillar\\data\\Si_pillar_radius_test2.txt'
raw = pd.read_table(fn, sep='\t', header=None)
# wavelength, Transmission
test_set_2 = torch.asarray(raw.values, dtype=torch.float32)
wavelength = test_set_2[:, 0]
radius = 0.15 * torch.ones_like(wavelength)
transmission2 = prediction(net, wavelength, radius)
with torch.no_grad():
    acc2 = eval_acc(transmission2, test_set_2[:, 1])
print(acc2)
ax = plt.subplot(1, 2, 2)
ax.plot(wavelength, transmission2.detach().numpy(),
        wavelength, test_set_2[:, 1])
# plt.show()
transmission2 = test_set_2[:, 1]

def spectrum_loss(x, wavelength=wavelength, target=test_set_1[:, 1]):
    radius = x * np.ones_like(wavelength)
    radius = torch.asarray(radius, dtype=torch.float32)
    X = torch.dstack((wavelength, radius)).reshape((wavelength.shape[0], 2))
    with torch.no_grad():
        y_hat = net(X)
    return ((y_hat.reshape(target.shape) - target) ** 2 / 2).mean().numpy()

# ls = spectrum_loss(0.21)

pso = PSO(dim=1, size=100, num_iter=1000, min_pos=[0.05], max_pos=[0.25],
          min_vel=[-0.1], max_vel=[0.1], tolerance=1e-7, loss=spectrum_loss, C1=2, C2=2, W=1)
fitnessList, positionList, bestPos = pso.update_ndim()
print('radius: %.8f' % (bestPos))