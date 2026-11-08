"""The native 128-point grid-sphere BPS, without optional CUDA extensions."""

import numpy as np
from scipy.spatial import cKDTree


def bps_basis(count=128, radius=0.2):
    # Match bps_torch's float32 cube and deterministic Fibonacci remainder.
    def grid(size):
        xyz = np.meshgrid(*([np.linspace(-1, 1, size)]*3))
        points = np.stack([v.ravel() for v in xyz], -1).astype(np.float32)*radius
        return points[np.linalg.norm(points, axis=1) < radius]

    size = int((6*count/np.pi)**(1/3))
    while len(grid(size)) < count:
        size += 1
    inside = grid(size-1)
    remaining = count-len(inside)
    i = np.arange(remaining)
    y = (i+0.5)*2/remaining-1
    phi = ((i+1) % remaining)*np.pi*(3-np.sqrt(5))
    r = np.sqrt(1-y*y)
    surface = np.stack((np.cos(phi)*r, y, np.sin(phi)*r), -1)*radius
    return np.concatenate((inside, surface)).astype(np.float32)


def encode_bps(vertices):
    return cKDTree(np.asarray(vertices, dtype=np.float32)).query(bps_basis())[0].astype(np.float32)
