import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from scipy.misc import imresize
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def data_gen_small(img_dir, mask_dir, lists, batch_size, dims, n_labels):
    

    # Loading the training data
    PATH = os.getcwd()
    # Define data path
    data_path = img_dir
    mask_path = mask_dir
    images = os.listdir(data_path)    
    masks = os.listdir(mask_path)
    #print(images,masks)
       
        
    imgs = []
    labels = []
    #images = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]
    #masks = [img for img in os.listdir(mask_dir) if img.endswith(".png")]
    #print ('Loaded the images of dataset-'+'{}\n'.format(dataset)) 
        
    for i,dataset in enumerate(images):
        # images
        img_path = data_path + '/'+ dataset
        #print(img_path)
        img = image.load_img(img_path, target_size=(480, 360))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = x/255
        #print('Input image shape:', x.shape)
        
        
        imgs.append(x)
        # mask
        
        mask_path = mask_dir + '/'+ masks[i]
        #print(mask_path)
        original_mask = image.load_img(mask_path, target_size=(480, 360))
        x1 = image.img_to_array(original_mask)
        x1 = x1/255
        x1 = x1[:,:,0]
            #print('Input mask shape:', x1.shape)
            #resized_mask = cv2.resize(original_mask, (dims[0], dims[1]))
            #print(resized_mask[:,:,0].shape)
        array_mask = catelab(x1, dims, n_labels)
        labels.append(array_mask)
    imgs = np.array(imgs)
    #print (imgs.shape)
    imgs=np.rollaxis(imgs,1,0)
    #print (imgs.shape)
    imgs=imgs[0]
    #print (imgs.shape)
    labels = np.array(labels)
    #print(labels.shape)
    return imgs, labels
    

#white 1 black 0
def catelab(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    #print(dims[0],dims[1])
    for i in range(dims[0]):
        for j in range(dims[1]):
            if(labels[i][j]>0):
                labels[i][j]=1
            #print(labels[i][j])
            x[i, j, int(labels[i][j])]=1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x
