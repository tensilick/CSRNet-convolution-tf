

# Borrowed from https://github.com/leeyeehoo/CSRNet-pytorch
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import scipy
import json
from matplotlib import cm as CM
from tqdm import tqdm
from numba import cuda
#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
# @cuda.autojit()
def gaussian_filter_density(gt):
#     print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(np.c_[np.nonzero(gt)[1], np.nonzero(gt)[0]])
    
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

#     print ('generate density...')
    for i, pt in (enumerate(pts)):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1: