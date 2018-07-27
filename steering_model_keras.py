##############################################################################
##
## steering_model_keras.py
##
## @author: Matthew Cline
## @version 20180724
##
## Description: Imiplementation of the steering model in Keras as opposed to 
## raw tensorflow code.
##
##############################################################################

import os
from time import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
import argparse

from keras.models import Model, Sequential, model_from_json, load_model
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from vis.visualization import visualize_saliency, visualize_cam, overlay

parser = argparse.ArgumentParser(description="Steering Model")
parser.add_argument('--model', type=str, help='path to the model checkpoints', default='models/keras')
parser.add_argument('--action', type=str, help='Action to perform on the selected model, current list: {train, continue, test}', default='continue')
parser.add_argument('--iters', type=int, help='Max number of iterations to train the model', default=25)
parser.add_argument('--batch', type=int, help='Batch size to use in training', default=10)
parser.add_argument('--init_epoch', type=int, help='Initial epoch to load the model from', default=1)
args = parser.parse_args()

cur_model = args.model
user_epochs = args.iters
user_batch = args.batch
user_init_epoch = args.init_epoch
action = args.action


def build_cnn(image_size=None, weights_path=None):
    image_size = image_size or (240, 320)
    if K.image_dim_ordering() == 'th':
        input_shape=(3,) + image_size
    else:
        input_shape = image_size + (3,)

    img_input = Input(input_shape)

    x = Convolution2D(filters=64, kernel_size=5, strides=1, activation='linear', padding='same')(img_input)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Convolution2D(filters=32, kernel_size=5, strides=1, activation='linear', padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Convolution2D(filters=16, kernel_size=3, strides=1, activation='linear', padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Convolution2D(filters=8, kernel_size=3, strides=1, activation='linear', padding='same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    y = Flatten()(x)
    y = Dense(256, activation='linear')(y)
    y = LeakyReLU()(y)
    y = Dropout(.5)(y)
    y = Dense(64, activation='linear')(y)
    y = LeakyReLU()(y)
    y = Dropout(.5)(y)
    y = Dense(1)(y)

    model = Model(inputs=img_input, outputs=y)
    model.compile(optimizer=Adam(lr=1e-4), loss = 'mse', metrics=['mse', 'mae', 'mape'])

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_images(imgs, test, hist, norm):
    imgList = []
    for imgID in imgs:
        if test:
            imgPath = os.path.normpath("Data/Test/Images/" + str(imgID) + ".jpg")
        else:
            imgPath = os.path.normpath("Data/Train/Images/" + str(imgID) + ".jpg")
        if hist:
            tempImg = cv2.imread(imgPath)
            tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2YUV)
            tempImg[:,:,0] = cv2.equalizeHist(tempImg[:,:,0])
            tempImg = cv2.cvtColor(tempImg, cv2.COLOR_YUV2RGB)
        else:
            tempImg = cv2.imread(imgPath)
            tempImg = cv2.cvtColor(tempImg, cv2.COLOR_BGR2RGB)
        if norm:
            tempImg = tempImg / 255.0
        tempImg = cv2.resize(tempImg, (320, 240))
        imgList.append(tempImg)
    return np.array(imgList)


def data_loader(imgs, angles, batch_size, test, hist, norm, pred=False):
    L = len(imgs)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)
            X = load_images(imgs[batch_start:limit], test, hist, norm)
            Y = angles[batch_start:limit]

            if pred:
                yield(X)
            else:
                yield (X,Y)

            batch_start += batch_size
            batch_end += batch_size

### Read in the data ###
trainImages = pickle.load(open("trainImagesShuffled.p", "rb"))
trainAngles = pickle.load(open("trainAnglesShuffled.p", "rb"))
valImages = pickle.load(open("valImagesShuffled.p", "rb"))
valAngles = pickle.load(open("valAnglesShuffled.p", "rb"))
testImages = pickle.load(open("testImages.p", "rb"))
testAngles = pickle.load(open("testAngles.p", "rb"))

tensorboard = TensorBoard(log_dir="models/keras/logs", histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
model_checkpoint = ModelCheckpoint('models/keras/steering_model-{epoch}.hdf5', verbose=1, save_best_only=False)

if action == 'train':

    steering_model = build_cnn()

    history = steering_model.fit_generator(
        generator=data_loader(trainImages, trainAngles, batch_size=user_batch, test=False, hist=True, norm=True),
        validation_data=(load_images(valImages[20:45], test=False, hist=True, norm=True), valAngles[20:45]),
        steps_per_epoch=len(trainImages)/user_batch,
        epochs=user_epochs,
        verbose=1,
        callbacks=[tensorboard, model_checkpoint])

    steering_model.save_weights(os.path.normpath("models/keras/model_bak.h5"))

    pickle.dump(history, open(os.path.normpath("models/keras/model_history.p"), "wb"))

if action == 'continue':

    steering_model = load_model(os.path.normpath(cur_model + "/steering_model-" + str(user_init_epoch) + ".hdf5"))
    history = steering_model.fit_generator(
        generator=data_loader(trainImages, trainAngles, batch_size=user_batch, test=False, hist=True, norm=True),
        validation_data=(load_images(valImages[20:45], test=False, hist=True, norm=True), valAngles[20:45]),
        steps_per_epoch=len(trainImages)/user_batch,
        epochs=user_epochs,
        verbose=1,
        callbacks=[tensorboard, model_checkpoint],
        initial_epoch=user_init_epoch)
    pickle.dump(history, open(os.path.normpath("models/keras/model_history_cont.p"), "wb"))

if action == 'test':
    steering_model = load_model(os.path.normpath(cur_model + "/steering_model-" + str(user_init_epoch) + ".hdf5"))
    pred = steering_model.predict_generator(
        generator=data_loader(testImages, testAngles, batch_size=user_batch, test=True, hist=True, norm=True, pred=True),
        steps=len(testImages)/user_batch,
        verbose=1).flatten()
    pred[pred > 1.0] = 1.0
    pred[pred < -1.0] = -1.0
    err = pred - testAngles
    mse = np.mean(err**2)
    rmse = np.sqrt(mse)
    print("MSE: %f, RMSE: %f\n\n" % (mse, rmse))
    print("Preparing the error distribution histogram...\n\n")
    plt.figure()
    plt.hist(np.array(err), bins=100)
    plt.show()
    indx = np.arange(len(testImages))
    print("Plotting predictions vs ground truth...\n\n")
    plt.figure()
    plt.plot(indx, pred, 'b')
    plt.plot(indx, testAngles, 'r')
    plt.show()


if action == 'saliency':
    steering_model = load_model(os.path.normpath(cur_model + "/steering_model-" + str(user_init_epoch) + ".hdf5"))
    titles = ['Left Steering', 'Right Steering', 'Maintain Steering']
    modifiers = ['negate', None, 'small_values']
    # titles = ["Saliency"]
    # modifiers = [None]
    img_collection = load_images(testImages[528:540], test=True, hist=True, norm=True)
    img = img_collection[8]
    for i, modifier in enumerate(modifiers):
        heatmap = visualize_cam(steering_model, layer_idx=-1, filter_indices=0,
            seed_input=img, grad_modifier=modifier)
        plt.figure()
        plt.title(titles[i])
        plt.imshow(overlay(img, heatmap, alpha=0.7))
    plt.figure()
    plt.title("Original Image")
    plt.imshow(img)
    plt.show()

if action == 'info':
    steering_model = load_model(os.path.normpath(cur_model + "/steering_model-" + str(user_init_epoch) + ".hdf5"))
    print(steering_model.summary())