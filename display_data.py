# Loading Dicom files Shoulder Data

import numpy as np
import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt
from skimage import filters
from skimage.transform import resize
import glob
import os
import cv2
from skimage import io, color
from skimage.measure import label, regionprops, regionprops_table

    
data_train = np.load('data_train_b_full.npy')
data_mask_train = np.load('data_mask_train_b_full.npy')
data_test = np.load('data_test_b_full.npy')
data_mask_test = np.load('data_mask_test_b_full.npy')



for i in range(0, data_train.shape[0]): 
    
    
    plt.subplot(1, 2, 1)
    plt.imshow(data_train[i,:,:],cmap='gray')
    plt.title("US Image")
    plt.subplot(1, 2, 2)
    plt.imshow(data_mask_train[i,:,:],cmap='gray')
    plt.title('GT Mask')
    plt.pause(0.5)
    plt.show()







