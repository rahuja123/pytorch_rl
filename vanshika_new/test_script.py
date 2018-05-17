#############TEST IMAGE SCRIPT ###############

import os
import glob
import numpy as np
import random   


root = 'viper_output/'

files = ['cam_a_test']
pos = 0
neg = 1
first_image_number= 51
second_image_number = 16    
anchor_path =  'person_{:04d}'.format(first_image_number)
second_image_number_= 'person_{:04d}'.format(second_image_number)
print anchor_path


pairs = []
complete_anchor_path =  root+files[0]+'/'+str(anchor_path)


##POSITIVE
if pos:
    a = sorted(glob.glob(complete_anchor_path+'/*.png'))
    positive_path = root+'cam_b_test/'+str(second_image_number_)
    b = sorted(glob.glob(positive_path+'/*.png'))
    # b
    positive_pair = []
    ## Appending Region Wise for Positive Pair
    for i in range(0,3):
        temp = []
        temp.append(a[i])
        temp.append(b[i])
        positive_pair.append(temp)
    positive_pair.append(1)
    pairs.append(positive_pair)     




##NEGATIVE
if neg:
    a = sorted(glob.glob(complete_anchor_path+'/*.png'))
    negative_path = root+'cam_b_test/'+str(second_image_number_)
    c = sorted(glob.glob(negative_path+'/*png'))
    negative_pair = []

        ## Appending Region Wise for Negative Pair
    for i in range(0,3):
        temp = []
        temp.append(a[i])
        temp.append(c[i])
        negative_pair.append(temp)    
    negative_pair.append(0)
    pairs.append(negative_pair)   

print pairs[0]
pairs = np.asarray(pairs)
new_pairs = np.random.permutation(pairs)



img_list = np.load('viper_images_train_ref.npy')
im = img_list[2*int(first_image_number)][0]

image = []
images = []
image.append(im)
image.append(int(first_image_number))
#print im,image[0]
image = np.array(image)
images.append(image)
im2 = img_list[2*int(second_image_number)+1][0]
image = []
image.append(im2)
image.append(int(second_image_number))
image = np.array(image)
images.append(image)
np.save('test_image',images)

print len(new_pairs)
np.save('testing_pairs',new_pairs)
"""
        im = 'VIPeR/cam_a/000_45.bmp'
        index = 0
        image = []
        images = []
        image.append(im)
        image.append(index)
        image = np.array(image)
        images.append(image)
        print images
        np.save('test_image',images)
"""