

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 09:24:49 2016

@author: bas
"""

from PIL import Image
import multiprocessing
from multiprocessing import Pool
import numpy as np
import glob
import fnmatch

import os
import shutil

source_root = '/Volumes/DISKIMAGES/Im_256/'
process_root = '/Volumes/DISKIMAGES/Proc_256_05/'
#source_dir_list = ['450D/','600D/','D600/',\
#'D90/','SigmaDPMerrill/','galaxy/',\
#'5DMK2/','6D/'	,'D800/','M9/'\
#	,'alphaA7/','iphone/']
#source_dir_list = ['5DMK2/']
crop_size = 256, 256


def generate_process(im_name):
    try:
        print im_name
        pil_image = Image.open(im_name)
    
        # convert to luminance
        im_size = pil_image.size
        sigma = 2
        im_array = np.asarray(pil_image).astype(float)
        noise = np.random.randn(im_size[0],im_size[1])*sigma
        im_array = im_array + noise
        im_array = np.round(im_array)
        im_array[im_array<0]=0
        im_array[im_array>=255]=255
        
        # create appropriate dirrectorries
        dir_model = os.path.split(os.path.split(im_name)[0])[0]
        dir_model = dir_model.replace(source_root,process_root)
        #print dir_model
        if not os.path.exists(dir_model):
            os.makedirs(dir_model)
        dir_block = os.path.split(im_name)[0]
        dir_block = dir_block.replace(source_root,process_root)
        #print dir_block
        if not os.path.exists(dir_block):
            os.makedirs(dir_block)
        im_out_name = im_name.replace(source_root,process_root)
        im_array = im_array.astype(dtype=np.uint8)
        im_noise_pgm = Image.fromarray(im_array)
        im_noise_pgm.save(im_out_name)
             
    except:
        pass
#                            

if __name__ == "__main__":
    list_im = []
    for root, dirnames, filenames in os.walk(source_root):
        for filename in fnmatch.filter(filenames, '*.pgm'):
            list_im.append(os.path.join(root, filename))
 
    #print list_im[:10]
    if not os.path.exists(process_root):
        os.makedirs(process_root)    
    nbCores = multiprocessing.cpu_count()
    pool = Pool(nbCores)
    pool.map(generate_process, list_im)
    pool.close()
    pool.join()
#    generate_process(list_im[0])













