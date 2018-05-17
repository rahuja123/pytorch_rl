
import os
import glob
import numpy as np

root = 'output_images/'
files = os.listdir(root)
files = ['cam_a','cam_b']

image_paths = []
labels = []

person_path = []
images = []

for f in files:
    #print f
    for i in xrange(2,201):
        person_path =  root+f+'/person_{:04d}'.format(i)
        #print person_path
        #a = glob.glob(person_path+'/*.png')
        #print a
        a = person_path + '/region1.jpg'
        for j in range(0,len(a)):
            image = []
            image.append(a[j])
            image.append(i-1)
            images.append(image)

images = np.asarray(images)
print images[0]
print len(images)
#np.save('prid_head',images)