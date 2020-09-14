import os
import functools
from pathlib import Path
import cv2
import numpy as np
from src.image_np import dct2,load_image
def convert_images_dct(inputs, tf=False, grayscale=True, log_scaled=False, abs=False):
    image= inputs
    image = dct2(image)
    if log_scaled:
        image = np.abs(image)
        image += 1e-12
        image = np.log(image)

    if abs:
        image = np.abs(image)

    return image
filedir = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/img/train/B_fake/'
outdir = '/home/liangyf/env/DCTAnalysis/ws/DCTAnalysis/data/FF/c23/all/dct/train/B_fake/'
for filename in os.listdir(filedir):
    #print (filename)
    imgdir = filedir + filename
    img = cv2.imread(imgdir)
    img_dct = convert_images_dct(img)
    dct_dir = outdir + filename
    cv2.imwrite(dct_dir,img_dct)
