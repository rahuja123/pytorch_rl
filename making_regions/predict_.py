import numpy as np
import argparse
import tensorflow as tf
from skimage.io import imsave
from PIL import Image
from os.path import join
import scipy.misc
import time
from skimage.transform import rescale, resize, downscale_local_mean
from keras import backend as K
from keras.models import load_model

from model.util import preprocess_input
from model.loss import per_pixel_softmax_cross_entropy_loss, IOU
from matplotlib import pyplot as plt
import os

custom_objects_dict = {
    'per_pixel_softmax_cross_entropy_loss': per_pixel_softmax_cross_entropy_loss,
    'IOU': IOU
}

name_counter = 0

def parse_arguments_from_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",
            help="Debug mode: more verbose, test things with less data, etc.",
            action='store_true')
    parser.add_argument("--demo",
            help='Demo the segmentation',
            action='store_true')
    parser.add_argument("--load_path",
            help="optional path argument, if we want to load an existing model")
    args = parser.parse_args()
    return args

# https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
def load_image(path):
    img = Image.open(path)
    oldsize = img.size

    img = img.resize((224, 224), Image.ANTIALIAS)
    img.load()
    data = np.asarray(img, dtype='int32')

    return data, oldsize

def save_image(array, path, size):
    img = Image.fromarray((array).astype(np.uint8), 'RGB')
    # Reshape it back up to the original size, since we shrunk to (224,224) at first
    img = img.resize((299,299), Image.ANTIALIAS)
    img.save(path)


if __name__ == "__main__":
    
    args = parse_arguments_from_command()
    demo = args.demo
    stored_model_path = 'weights/diamondback_ep11-tloss=34423.1392-vloss=44111.4496-tIOU=0.7913-vIOU=0.7554.h5'

    if stored_model_path is None:
        stored_model_path = input("Load model from rel path: ")

    model = load_model(stored_model_path, custom_objects=custom_objects_dict)

    while demo: # If demo is true, keep doing this loop.
        #filename = input("Place image in demo-images. Image file name w/ extension?: ")
        filename= "person_0004.png"
        array, oldsize = load_image(join("demo-images", filename))
        original_image= array

        array = array.astype(np.float64)
        array = array[np.newaxis, ...]
        array = preprocess_input(array)

        y = model.predict(array, verbose=1)
        y = np.argmax(y, axis=-1)
        y = np.squeeze(y) # removebatch dim
        y_3d= np.array([y, y, y])

        new_rgb= np.multiply(y_3d.transpose(1,2,0), original_image)



        img_head = Image.fromarray((head).astype(np.uint8), 'RGB').resize(oldsize, Image.ANTIALIAS)
        img_body = Image.fromarray((body).astype(np.uint8), 'RGB').resize(oldsize, Image.ANTIALIAS)
        img_torso = Image.fromarray((torso).astype(np.uint8), 'RGB').resize(oldsize, Image.ANTIALIAS)

        # Reshape it back up to the original size, since we shrunk to (224,224) at first
        img_head.save(join("demo-images", filename + "-rgb-head.jpg"))
        img_body.save(join("demo-images", filename + "-rgb-body.jpg"))
        img_torso.save(join("demo-images", filename + "-rgb-torso.jpg"))
        exit()

        #scipy.misc.toimage(new_rgb, cmin=0.0, cmax=...).save('outfile.jpg')
        #save_image(new_rgb, join("demo-images", filename + "-rgb.jpg"), size=oldsize)


        #save_image(y, join("demo-images", )filename+"-pred.jpg"), size=oldsize)


    #########################
    # Processing images for putting on the poster, predicting on all these files.

    for k in range(200):

        input_dir = "/home/vanshika/Desktop/MXnet_RL_Code - Copy/rl-multishot-reid-master/prid_2011/single_shot/cam_b"
        output_dir = "assets/output_imgs__/cam_b"

        array, oldsize = load_image(join(input_dir, "person_%04d.png" %(k+1)))
        original_image= array

        # #plt.figure()
        #plt.imshow(original_image)
        #plt.show()

        array = array.astype(np.float64)
        array = array[np.newaxis, ...] # add batch dim
        array = preprocess_input(array) # preprocess for DenseNet
        #plt.figure()
        #plt.imshow(original_image)
        #plt.show()

        y = model.predict(  array, verbose=1)

        y = np.argmax(y, axis=-1)
        y = np.squeeze(y) # remove batch dim
        print (y.shape)
        #plt.figure()
        #plt.imshow(y)
        #plt.show()
        #print (y[0:20])
        y_3d= np.array([y, y, y])
        #y_3d = np.transpose(y_3d,(1,2,0))


        #new_rgb= np.multiply(y_3d.transpose(1,2,0), original_image)
        new_rgb = y
        print (new_rgb.shape)
        top_row = 0
        bottom_row = 0
        left_col = 0
        right_col = 0

        for i in range(224):
            for j in range(224):
                if new_rgb[i][j].any()!=0:
                    top_row = i
                    break;

        for i in range(223,0,-1):
            for j in range(224):
                if new_rgb[i][j].any()!=0:
                    bottom_row = i
                    break;

        for j in range(224):
            for i in range(224):
                if new_rgb[i][j].any()!=0:
                    left_col = j
                    break;

        for j in range(223,0,-1):
            for i in range(224):
                if new_rgb[i][j].any()!=0:
                    right_col = j
                    break;

        print (top_row)
        print (bottom_row)
        print (left_col)
        print (right_col)
        print (original_image.shape)
        #plt.figure()
        #plt.imshow(original_image)
        
        #plt.show()
        if top_row==0 and bottom_row==0:
            bottom_row = 0
            top_row = 224
        if right_col==0 and left_col==0:
            right_col = 0
            left_col = 224
        
        original_image = original_image[bottom_row:top_row,right_col:left_col,:]
        original_image = Image.fromarray((original_image).astype(np.uint8), 'RGB')
        
        original_image = original_image.resize((224,224), Image.ANTIALIAS)

        #print (original_image.shape)

        original_image = np.asarray(original_image)
        head= original_image[0:45, :, :]
        torso= original_image[46: 120, :, :]
        leg= original_image[121:-1, :, : ]
        final_dir = join(output_dir, "person_%04d" %(k+1))

        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        #save_image(head, join(output_dir,  "person_%04d-head.jpg" % (i)), size=oldsize)
        #save_image(torso, join(output_dir,  "person_%04d-torso.jpg" % (i)), size=oldsize)
        #save_image(leg, join(output_dir,  "person_%04d-leg.jpg" % (i)), size=oldsize)
        save_image(head, join(final_dir,  "region1.jpg" ), size=oldsize)
        save_image(torso, join(final_dir,  "region2.jpg") , size=oldsize)
        save_image(leg, join(final_dir,  "region3.jpg" ), size=oldsize)
        #save_image(original_image, join(final_dir,  ".jpg" ), size=oldsize)
        print("done with %d" %(k+1))
