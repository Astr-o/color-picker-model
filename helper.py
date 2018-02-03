import matplotlib.image as mpimg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import patches
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from PIL import Image
from scipy.spatial import distance

# center crop
def crop_img(img, crop, display=True):
    '''center crop image'''
    (w, h) = img.size
    
    (w_crop, h_crop) = crop
    
    w_pixel = w_crop * w 
    h_pixel = h_crop * h
    
    left = (w - w_pixel) // 2
    top = (h - h_pixel) // 2
    right = (w + w_pixel) // 2
    bottom = (h + h_pixel) // 2
    
    img = img.crop((left, top, right, bottom))
    
    return img

def plot_img(img, title='', figsize=(5,5)):
    img = np.asarray(img)
    
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(title)
    plt.show()

def plot_swatches(colors):
    fig = plt.figure()

    plt.axis('off')
    ax = fig.add_subplot(111)

    x_from = 0.00
    
    n = len(colors)
    
    for c in colors:
        ax.add_patch(patches.Rectangle( (x_from, 0.05), ((1/n) - 0.02), 0.9, alpha=None, facecolor=tuple(c), edgecolor='black') )
        x_from = x_from + (1/n)

    fig.show()
    plt.show()

def get_distance(x,y):
    
    if len(x) == 4:
        x = (x[0], x[1], x[2])
        
    if len(y) == 4:
        y = (y[0], y[1], y[2])
    
    return distance.euclidean(x,y)

def get_closest_point(p, pxs):
    d_min = None
    p_min = None
    
    for px in pxs:
        d = get_distance(p, px)
        
        if d_min is None or d < d_min:
            d_min = d
            p_min = px
        
    return p_min

def get_closest_color(c, colormap):
    named_c = get_closest_point(c, list(colormap.keys()))  
    return colormap[named_c]
    
def get_html_color_dict():
    safe = mcolors.CSS4_COLORS

    safe_map = { hex_to_rgb(v):k for (k,v) in safe.items()}

    return safe_map

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    rgb_int = tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3)) ## 255, 255, 255
    rgb_fl = tuple(i / 255 for i in rgb_int)
    
    return rgb_fl