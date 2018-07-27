import argparse
import base64
from datetime import datetime
import os
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import cv2

from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard

sio = socketio.Server()
app = Flask(__name__)

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED


def crop_image(image):
    return image[15:-25,80:-80,:] #remove the car from the image


def resize_image(image, height=240, width=320):
    return cv2.resize(image, (width, height), cv2.INTER_CUBIC)


def hist_eq(image):
    tempImg = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    tempImg[:,:,0] = cv2.equalizeHist(tempImg[:,:,0])
    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_YUV2RGB)
    return tempImg


def normalize_image(img):
    tempImg = img / 255.0
    return tempImg


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            global hist
            global norm
            image = np.asarray(image)
            image = crop_image(image)
            image = resize_image(image)
            if hist:
                image = hist_eq(image)
            if norm:
                image = normalize_image(image)
            image = np.array([image])
            steering_angle = model.predict(image)[0,0]
            scaledAngle = steering_angle * -2.0
            print("Steering Angle Prediction: ", steering_angle)
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - scaledAngle**2 - (speed/speed_limit)**2
            print("Steering Angle: %f\nThrottle: %f\n" % (scaledAngle, throttle))
            send_control(scaledAngle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--model',
        type=str,
        help='The model location',
        default='models/keras')
    # parser.add_argument(
    #     '--best',
    #     type=int,
    #     help='Whether or not to use the best epoch',
    #     default=True)
    parser.add_argument(
        '--epoch',
        type=int,
        help='Epoch to load for the model',
        default=5)
    parser.add_argument(
        '--hist',
        type=int,
        help='Perform histogram equalization on the images, 0=No, 1=Yes',
        default=0)
    parser.add_argument(
        '--norm',
        type=int,
        help='Perform normalization on the input images, 0-No, 1=Yes',
        default=0)
    parser.add_argument(
        '--weights',
        type=str,
        help="Location of the weights for the keras model.",
        default='models/keras/model.h5')
    args = parser.parse_args()

    hist = args.hist
    norm = args.norm
    init_epoch = args.epoch
    model_dir = args.model

    if hist:
        print("Histogram equalization enabled.")
    if norm:
        print("Image normalization enabled.")

    model = load_model(os.path.normpath(model_dir + "/steering_model-" + str(init_epoch) + ".hdf5"))
    
    app = socketio.Middleware(sio, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
