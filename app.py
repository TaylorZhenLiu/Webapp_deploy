# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import scipy.misc as misc

import matplotlib as plt
import numpy as np
import tensorflow as tf

from regressionnet2_upsmapling2 import RegressionNet
from finetune_mian_diff_initial_classify import DeepLabLFOVModel
from finetune_VGG_deeplab_seg import DeepLabSEGModel

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


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# default access page
@app.route("/")
def main():
    return render_template('index.html')

# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".mp4"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("index_loaded.html", image_name=filename)

def read_image_from_disk(img_filename):
    img3 = np.zeros((321,321,3))
    img = misc.imread(img_filename)
    img = misc.imresize(img,[321,321])
    img = img.astype("float32")
    max_ = np.amax(img)
    min_ = np.amin(img)
    img = 255*(img - min_) / (max_ - min_)
    img3[:,:,0]=img
    img3[:,:,1]=img
    img3[:,:,2]=img
    return img3.astype("float32")

def image_slice(image_batchs, index):
    image_s = np.squeeze(image_batchs[index,:,:,:], axis=0)
    return image_s

@app.route("/segment", methods=["POST"])
def segment():
    filename = request.form['image']
    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])
    img = read_image_from_disk(destination)
    org_img = img
    image_batch = np.zeros((1, 321, 321, 3))
    image_batch[0,:,:,:] = img
    image_batch = np.reshape(image_batch, (batch_size, 321, 321, 3))
    ind = tf.placeholder(tf.int32, shape=(1, 1))
    img_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32)
    print("img_batch" + repr(img_batch))

    img_slice = tf.py_func(image_slice, [img_batch, ind], tf.float32)

    net_deeplab = DeepLabLFOVModel(WEIGHTS_PATH)
    net_regression = RegressionNet()
    net_seg = DeepLabSEGModel(WEIGHTS_PATH)

    _, _, _, train_feature1024 = net_deeplab.preds(img_slice)
    trainpred_regression = net_regression.preds(train_feature1024)
    trainpred_input = tf.concat([trainpred_regression * 255, trainpred_regression * 255, trainpred_regression * 255], 3)
    trainmask_segmentation, _, _, _ = net_seg.preds(trainpred_input)
    # Which variable to load
    trainable = tf.trainable_variables()
    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    # Load weights
    saver = tf.train.Saver(var_list=trainable)
    # load(saver, sess, model_weights)
    saver.restore(sess, model_weights)
    # Perform inference
    # if not os.path.exists(label_dir):
    #    os.makedirs(label_dir)
    preds = sess.run([trainmask_segmentation], feed_dict={ind: np.reshape(0, (1, 1))})
    img = np.asarray(preds[0][0] * 255)
    img = Image.fromarray(img[:, :, 0])

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    # return send_image('temp.png')
    return render_template("index_segmented.html", image_name=filename, image_name_mask='temp.png')

@app.route("/clear", methods=["POST"])
def clear():
    return render_template('index.html')


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()

