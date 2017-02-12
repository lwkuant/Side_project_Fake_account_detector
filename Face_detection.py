# -*- coding: utf-8 -*-
"""
Face Detection
"""

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt 
%matplotlib inline 
import numpy as np

from skimage import data, color, feature
import skimage.data
image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualise=True)
fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')
ax[1].imshow(hog_vis)
ax[1].set_title('visualization of HOG features');

# obtain the positive training dataset
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
print(positive_patches.shape)

# obtain the negative training dataset
