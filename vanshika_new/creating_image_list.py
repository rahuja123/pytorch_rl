
import os
import glob
import numpy as np

root = 'data/'
files = os.listdir(root)
#print files
files = ['train']
image_paths = []
labels = []

person_path = []
images = []

for f in files:
    #print f
    for i in xrange(0,632):
        person_path =  root+f+'/'+ str(i)

        a = glob.glob(person_path+'/*.png')
        print a
        
        for j in range(0,len(a)):
            image = []
            image.append(a[j])
            image.append(i)
            images.append(image)

images = np.asarray(images)
#images = np.random.permutation(images)
print images[0:10]
np.save('viper_images_train_ref',images)