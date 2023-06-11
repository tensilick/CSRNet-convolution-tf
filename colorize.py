# from https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b
import matplotlib
import matplotlib.cm
import numpy as np
import tensorflow as tf

def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it 