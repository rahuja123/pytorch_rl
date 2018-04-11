
import os
import glob
import numpy as np

root = '../../prid_2011/multi_shot/'
files = os.listdir(root)
#print files

image_paths = []
labels = []

person_path = []
images = []

for f in files:
    #print f
    for i in xrange(101,151):
        person_path =  root+f+'/person_{:04d}'.format(i)
        #print person_path
        a = glob.glob(person_path+'/*.png')
        #print a
        
        for j in range(0,len(a)):
            image = []
            image.append(a[j])
            image.append(i-1)
            images.append(image)

images = np.asarray(images)
#print images[0]
np.save('test_images',images)