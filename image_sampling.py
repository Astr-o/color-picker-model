""" this module contains methods that select sample pixels from an image """
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist

def rand_sampling(width=1, height=1, n=500):
    sample = [(np.random.uniform(0, width), np.random.uniform(0, height)) for _ in range(0, n) ]
    
    return sample

def rand_sample_img_array(array, plot=False):
    w, h, d = tuple(array.shape)
    
    print(w,h,d)
    
    sample = rand_sampling(width=w, height=h, n=500)
    sample = np.vectorize(lambda x: int(round(x)))(sample)
    
    sample_pixels = [array[x,y] for (y,x) in sample if x >= 0 and x <= w - 1 and y >= 0 and y <= h - 1]
    
    if plot:
        df_sample = pd.DataFrame(sample, columns=['y', 'x'])
        plot = sns.jointplot(x="x", y="y", data=df_sample)
        
        
    return (sample_pixels, df_sample)

## based on https://www.labri.fr/perso/nrougier/from-python-to-numpy/code/DART_sampling_numpy.py
def DART_sampling_numpy(width=1.0, height=1.0, radius=0.025, k=100):

    # Theoretical limit
    n = int((width+radius)*(height+radius) / (2*(radius/2)*(radius/2)*np.sqrt(3))) + 1
    # 5 times the theoretical limit
    n = 5*n

    # Compute n random points
    P = np.zeros((n, 2))
    P[:, 0] = np.random.uniform(0, width, n)
    P[:, 1] = np.random.uniform(0, height, n)

    # Computes respective distances at once
    D = cdist(P, P)

    # Cancel null distances on the diagonal
    D[range(n), range(n)] = 1e10

    points, indices = [P[0]], [0]
    i = 1
    last_success = 0
    while i < n and i - last_success < k:
        if D[i, indices].min() > radius:
            indices.append(i)
            points.append(P[i])
            last_success = i
        i += 1
    return points

def dart_sample_img_array(array, plot=False):
    w, h, d = tuple(array.shape)
    
    print(w,h,d)
    
    sample = DART_sampling_numpy(width=w, height=h, radius=5, k=100)
    sample = np.vectorize(lambda x: int(round(x)))(sample)
    
    sample_pixels = [array[x,y] for (y,x) in sample if x >= 0 and x <= w - 1 and y >= 0 and y <= h - 1]
    df_sample = pd.DataFrame(sample, columns=['y', 'x'])
    if plot:
        plot = sns.jointplot(x="x", y="y", data=df_sample)
        
        
    return (sample_pixels, df_sample)
    