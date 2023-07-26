import tensorflow as tf
import cv2
import numpy as np
from random import shuffle
import glob
import sys
import os

root = '/home/rishhanth/Documents/gen_codes/CSRNet-tf/ShanghaiTech/part_A/train_data/images/'

def get_filenames():
    filenames = os.listdir(root)
    image_files = []
    label_files = []
    for i in filenames:
        im_file = os.path.join(root,i)
        image_files.append(im_file)
        label_files.append(im_file.replace('IMG_','LAB_').replace('.jpg','.npy').replace('images','labels'))
    return image_files,label_files

shuffle_data = True  # shuffle the addresses before saving
# read addresses and labels from the 'train' folder
t