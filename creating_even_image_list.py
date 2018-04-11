
import os
import glob
import numpy as np
import random

root = '../../prid_2011/multi_shot/'
files = os.listdir(root)
#print files

image_paths = []
labels = []

person_path = []
images = []
for i in xrange(1,101):
    #print f
    a = []
    for f in files:
        person_path =  root+f+'/person_{:04d}'.format(i)
        #print person_path
        a.append(glob.glob(person_path+'/*.png'))
    
    length = len(a[0])  
    if len(a[0])>len(a[1]):
        length = len(a[1])

    for j in range(0,length):
            image = []
            image.append(a[0][j])
            image.append(i)
            images.append(image)
            image = []
            image.append(a[1][j])
            image.append(i)
            images.append(image)

images = np.asarray(images)
#print images[0]
#print images[1]
#print images[2]
#print images[3]
#print images[10000]
#print images[10001]
idx = range(0, len(images), 2)
random.shuffle(idx)
ret = []
for i in idx:
    ret.append(images[i])
    ret.append(images[i + 1])
#print ret[0]
#print ret[1]
#print ret[2]
#print ret[3]
#print ret[10000]
#print ret[10001]
ret = np.asarray(ret)    
np.save('images_even',ret)