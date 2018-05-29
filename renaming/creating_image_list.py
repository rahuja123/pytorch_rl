
import os
import glob
import numpy as np
from os.path import join


root = 'prid_output/'

#print files
files = ['cam_a','cam_b']
image_paths = []
labels = []

person_path = []
images = []
"""

###-----FOR RENAMING THE FOLDERS----------------###
files_a = sorted(os.listdir(join(root,files[0])))
files_b = sorted(os.listdir(join(root,files[1])))
#rint files_a
for f in range(len(files_b)):
    a = join(root,files[1],files_b[f])
    b = join(root,files[1],'person_{:04d}'.format(f))
    os.rename(a,b)
### ----- FOR REGIONS OF IMAGES --------- ###
"""
"""
for f in files:
    #print f
    for i in xrange(0,100):
        #person_path =  root+f+ '/person_{:04d}'.format(i) + '/region1.png'
        #person_path =  root+f+ '/person_{:04d}'.format(i) + '/region1.png'
        person_path =  root+f+ '/person_{:04d}'.format(i) + '/region1.jpg'
        #a = glob.glob(person_path)
        print person_path
        
        #for j in range(0,len(a)):
        image = []
        image.append(person_path)
        image.append(i)
        images.append(image)

images = np.asarray(images)
images = np.random.permutation(images)
print images[0:10]
np.save('prid_head_train',images)

"""

### ----- FOR FULL BODY IMAGES --------- ###

for f in files:
    #print f
    #a = sorted(glob.glob(root+f+'/*.bmp'))
    for i in xrange(0,100):
        person_path =  root+f+ '/person_{:04d}'.format(i) + '/person_{:04d}.jpg'.format(i)
        #person_path =  a[i]
        #a = glob.glob(person_path)
        print person_path
        
        #for j in range(0,len(a)):
        image = []
        image.append(person_path)
        image.append(i)
        images.append(image)

images = np.asarray(images)
images = np.random.permutation(images)
print images[0:10]
np.save('prid_full_body',images)
