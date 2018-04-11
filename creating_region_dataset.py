import cv2
import os

root = 'output_imgs/'
files = os.listdir(root)
#print files

path = 'output_imgs/'
for f in files:
    #print f
    for i in range(0,200):
        person_number = '/person_{:04d}-pred'.format(i)
        person_path = root+f+person_number
        print(person_path)
        directory = path+f+person_number
        if not os.path.exists(directory):
            os.makedirs(directory)
        final_path = person_path+'.jpg'
        print(final_path)
        im = cv2.imread(final_path)
        print(im.shape)


        im1 = im[0:45]
        im1 = cv2.resize(im[:27], (50, 64))
        im2 = cv2.resize(im[27:70], (50, 64))
        im3 = cv2.resize(im[70:128], (50, 64))
        cv2.imwrite(directory+'/region1.jpg',im1)
        cv2.imwrite(directory+'/region2.jpg',im2)
        cv2.imwrite(directory+'/region3.jpg',im3)
