import os
import glob
import numpy as np
import random

root = 'output_imgs/' #input directory
files = ['cam_a']
#print files

anchor_names = []
for f in files:
    print(f)
    for i in range(0,200):
        anchor_path =  'person_{:04d}-pred'.format(i)
        anchor_names.append(anchor_path)
print(len(anchor_names))
pairs = []

for f in files:
    #print f
    for i in anchor_names:
        #print i
        anchor_path =  root+f+'/'+i
        positive_path = root+'cam_b/'+i
        #print anchor_path
        temp = random.choice(anchor_names)
        print(temp)
        while temp==i:
            temp = random.choice(anchor_names)

        negative_path = root+'cam_b/'+temp
        #print a,b
        a = sorted(glob.glob(anchor_path+'/*.jpg'))
        b = sorted(glob.glob(positive_path+'/*.jpg'))
        c = sorted(glob.glob(negative_path+'/*.jpg'))
        positive_pair = []
        negative_pair = []
        #print a

        ## Appending Region Wise for Positive Pair
        for i in range(0,3):
            temp = []

            temp.append(a[i])
            temp.append(b[i])
            positive_pair.append(temp)

         ## Appending Region Wise for Negative Pair
        for i in range(0,3):
            temp = []

            temp.append(a[i])
            temp.append(c[i])
            negative_pair.append(temp)


        positive_pair.append(1)
        negative_pair.append(0)

        pairs.append(positive_pair)
        pairs.append(negative_pair)

#print pairs[0]
pairs = np.asarray(pairs)
new_pairs = np.random.permutation(pairs)
print(len(new_pairs))
np.save('pairs',new_pairs)
