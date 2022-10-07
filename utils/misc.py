import math
import torch
from torch import nn
from einops import repeat
import numpy as np


def freq_to_wave(freq, is_radian=True):
    _freq_rad = 2 * math.pi * freq if not is_radian else freq
    return torch.hstack([torch.cos(_freq_rad), torch.sin(_freq_rad)])


def unsqueeze_at_the_end(x, n):
    return x[(...,) + (None,)*n]


def get_RTmat(theta, phi, gamma, w, h, dx, dy):
    d = np.sqrt(h ** 2 + w ** 2)
    f = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    # Projection 2D -> 3D matrix
    A1 = np.array([[1, 0, -w / 2],
                   [0, 1, -h / 2],
                   [0, 0, 1],
                   [0, 0, 1]])

    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta), -np.sin(theta), 0],
                   [0, np.sin(theta), np.cos(theta), 0],
                   [0, 0, 0, 1]])

    RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                   [0, 1, 0, 0],
                   [np.sin(phi), 0, np.cos(phi), 0],
                   [0, 0, 0, 1]])

    RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                   [np.sin(gamma), np.cos(gamma), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, f],
                  [0, 0, 0, 1]])
    # Projection 3D -> 2D matrix
    A2 = np.array([[f, 0, w / 2, 0],
                   [0, f, h / 2, 0],
                   [0, 0, 1, 0]])
    return np.dot(A2, np.dot(T, np.dot(R, A1)))
