import os
import glob
import numpy as np
import random


root = 'viper_output/'

files = ['cam_a_test']


anchor_names = []
for f in files:
    #print f
    for i in xrange(0,2):
        anchor_path =  'person_{:04d}'.format(i)
        anchor_names.append(anchor_path)
print anchor_names

pairs = []

for f in files:
    #print f
    for i in anchor_names:
        #print i
        anchor_path =  root+f+'/'+i
        #positive_path = root+'cam_b_test/'+i
        print anchor_path
        temp = random.choice(anchor_names)
        while temp==i:
            temp = random.choice(anchor_names)
        
        negative_path = root+'cam_b_test/'+temp
        #print a,b
        a = sorted(glob.glob(anchor_path+'/*.png'))
        #b = sorted(glob.glob(positive_path+'/*.png'))
        c = sorted(glob.glob(negative_path+'/*png'))
        #positive_pair = []
        negative_pair = []
        #print a

        ## Appending Region Wise for Positive Pair
        #for i in range(0,3):
        #    temp = []

        #    temp.append(a[i])
        #    temp.append(b[i])
        #    positive_pair.append(temp)

         ## Appending Region Wise for Negative Pair
        for i in range(0,3):
            temp = []

            temp.append(a[i])
            temp.append(c[i])
            negative_pair.append(temp)
        

        #positive_pair.append(1)
        negative_pair.append(0)
        
        #pairs.append(positive_pair)
        pairs.append(negative_pair)   

print pairs[0]
pairs = np.asarray(pairs)
new_pairs = np.random.permutation(pairs)
print len(new_pairs)
np.save('pairs_test_viper_positive',new_pairs)
