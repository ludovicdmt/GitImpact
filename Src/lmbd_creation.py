
# -*- coding: utf-8 -*-


import os
import glob
import random
import numpy as np

from PIL import Image

from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

#def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
#
#    #Histogram Equalization
#    #img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
#    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
#    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
#
#    #Image Resizing
#    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
#
#    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=1, # images are in black and white 
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=img.tostring())
        
train_lmdb = '/path/to/lmdb_train'
validation_lmdb = '/path/to/lmdb_validation'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

############## Read images #############################

cover = [img for img in glob.glob("/path/to/images/Im_256/*/*/*.pgm")]
gaussian = [img for img in glob.glob("/path/to/images/Proc_256/*/*/*.pgm")] #Altered images
print 'Nombre d images cover:' ,str(len(cover))
print 'Nombre d images gaussian:', str(len(gaussian))

############## Train test split ########################

all_images = cover
all_images.extend(gaussian)

print 'Total number of images:', str( len(all_images))
random.shuffle(all_images)

train_test_split = 0.8

split_index = int(len(all_images)*train_test_split)

train_data = all_images[0:split_index]

print 'Number of images in train', str(len(train_data))

test_data = all_images[split_index::]

print 'Number of images in test',str(len(test_data))

#Shuffle
random.shuffle(train_data)
random.shuffle(test_data)

############## LMDB creation ###########################

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
        try:
            img = Image.open(img_path) # read in black and white
        
        except IOError:
            print str(img_path), 'can not be read.'

        else:
            img = np.asarray(img)

        
            if 'Im_256' in img_path:
                label = 0
            else:
               label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
       # print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(test_data):
        if in_idx % 6 != 0:
            continue
        try:
            img = Image.open(img_path)
        
        except IOError:
            print str(img_path), 'can not be read.'

        else:
            img = np.asarray(img)


            if 'Im_256' in img_path:
                label = 0
            else:
                label = 1
            datum = make_datum(img, label)
            in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
       # print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'

