import os
from PIL import Image
import numpy as np
import tensorflow as tf
import notebook.nbextensions
import zipfile
from six.moves.urllib.request import urlretrieve 
import argparse
from datetime import datetime
import sys
import time
import matplotlib.pyplot as plt
from six.moves import cPickle
#from reader_aug_mask import Image_Reader
#from kid_reader import ImageReader
from regressionnet2_upsmapling2 import RegressionNet
from finetune_mian_diff_initial_classify import DeepLabLFOVModel
from finetune_VGG_deeplab_seg import DeepLabSEGModel
import scipy.io as io
import scipy.misc as misc
import scipy.ndimage as ndi
import math
#from foo import bwmorph_thin
import networkx as nx
#os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
TEST_PATH = './test.txt'
IMAGE_PATH = './'
WEIGHTS_PATH   = 'VGG_16.npy'
model_weights = './aug_VGG_multilossdeeplabmask_1/model.ckpt-20000'
batch_size = 1
INPUT_SIZE =(321,321)
#label_dir = './result/normal104/end-to-end/'
thred = np.exp(-1.0)
thred0 = np.exp(-2.0)
lamda=1.0



def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    

    
def mask_to_distance(mask_img):
    dislab1 = ndi.distance_transform_edt(mask_img)
    dislab2 = ndi.distance_transform_edt(1-mask_img)
    dislab = dislab1
    dislab[mask_img==0] = dislab2[mask_img==0]
    dislab = np.exp(-(dislab-1))
    return dislab.astype("float32")

def image_slice(image_batchs, index):
    image_s = np.squeeze(image_batchs[index,:,:,:], axis=0)
    return image_s

def read_pred_label_list(image_dir,data_list):
    f = open(data_list, 'r')
    images = []
    pred_regname = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(os.path.join(image_dir,image))
        pred_regname.append(mask)          
    return images, pred_regname   


def read_image_from_disk(img_filename,ii):
    img3 = np.zeros((321,321,3))
    str_convert = ''.join(img_filename[ii])
    img = misc.imread(str_convert)
    img = misc.imresize(img,[321,321])
    img = img.astype("float32")
    max_ = np.amax(img)
    min_ = np.amin(img)
    img = 255*(img - min_) / (max_ - min_)
    img3[:,:,0]=img
    img3[:,:,1]=img
    img3[:,:,2]=img
    return img3.astype("float32")

    
def main():
    "Create the model and start the evaluation process."
    #args = get_arguments()
    image_name, pred_regnames  = read_pred_label_list(IMAGE_PATH,TEST_PATH)
    image_batch = np.zeros((batch_size,321,321,3))
    trainpred_distanceB = np.zeros((batch_size,321,321))
    trainlabel_distance321B = np.zeros((batch_size,321,321))
    trainmask_batch = np.zeros((batch_size,321,321)) 
    for ii in range(batch_size):
         image_batch[ii,:,:,:]= read_image_from_disk(image_name,ii)
    image_batch = np.reshape(image_batch,(batch_size, 321, 321, 3)) 
    ind = tf.placeholder(tf.int32, shape=(1, 1))
    img_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    print("img_batch"+repr(img_batch))

    img_slice = tf.py_func(image_slice, [img_batch,ind], tf.float32)
    print("img_slice"+repr(img_slice))  
    #Create network.

    net_deeplab = DeepLabLFOVModel(WEIGHTS_PATH)     
    net_regression = RegressionNet()
    net_seg = DeepLabSEGModel(WEIGHTS_PATH)

    _,_,_,train_feature1024 = net_deeplab.preds(img_slice)
    trainpred_regression = net_regression.preds(train_feature1024)
    trainpred_input = tf.concat([trainpred_regression*255, trainpred_regression*255, trainpred_regression*255], 3)
    trainmask_segmentation,_,_,_ = net_seg.preds(trainpred_input)

    #Which variable to load
    trainable = tf.trainable_variables()
    #print('trainable'+repr(trainable)) 
    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Load weights
    saver = tf.train.Saver(var_list=trainable)
    #load(saver, sess, model_weights) 
    saver.restore(sess, model_weights)
        # Perform inference
    #if not os.path.exists(label_dir):
    #    os.makedirs(label_dir)
    #for ii in range(batch_size):
    preds = sess.run([trainmask_segmentation],feed_dict={ind: np.reshape(0,(1,1))})
        #print('10')
#        io.savemat(label_dir+''.join(pred_regnames[ii]), {'predmask': preds})

    # normalize
    img = np.asarray(preds[0][0] * 255)
    img = Image.fromarray(img[:,:,0])
    plt.imshow(img)
    print('1.0')

            
    
if __name__ == '__main__':
    main()