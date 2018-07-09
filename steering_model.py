##############################################################################
##
## steering_model.py
##
## @author: Matthew Cline
## @version: 20180529
##
## Description: End to end steering model for a self driving car. The only
## sensor available to the model is a forward facing camera. The model
## will used the images from the forward facing camera to predict the 
## appropriate steering angle using a convolutional neural network
## implemnted in Tensorflow.
##
##############################################################################

import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import sys
import pickle
import time
import argparse
import json
import shutil
import psutil
import subprocess


np.set_printoptions(suppress=True)

### Kill tensorboard if it was running from a previous run ###
PROCKILL = "tensorboard.exe"
for proc in psutil.process_iter():
    if proc.name() == PROCKILL:
        proc.kill()


##############################################################################
##
## HOUSEKEEPING STUFF
##
##############################################################################

####### PARSE COMMAND LINE ARGUMENTS #######
parser = argparse.ArgumentParser(description="Steering Model")
parser.add_argument('--model', type=str, help='Model to use, current list: {CNN[5533]_nodrop, CNN[5533]_drop, CNN[3333]_nodrop, CNN[3333]_drop}', default='CNN[5533]_drop')
parser.add_argument('--action', type=str, help='Action to perform on the selected model, current list: {train, continue, test}', default='continue')
args = parser.parse_args()

cur_model = args.model
action = args.action

####### LOAD THE MODEL INFORMATION FROM THE MODEL TRACKER #######
model_tracking = json.load(open("model_tracker.json", "r"))
model_path = model_tracking[cur_model]["path"]
start_epoch = model_tracking[cur_model]["epoch"]
keep_rate = model_tracking[cur_model]["keep_rate"]

####### IF TRAIN IS SELECTED RESET THE MODEL TO ITS INITIAL STATE ######
if action == 'train':
    model_tracking[cur_model]["epoch"] = 0
    model_tracking[cur_model]["train_rmse"] = 100
    model_tracking[cur_model]["val_rmse"] = 100
    model_tracking[cur_model]["best_epoch"] = 0
    model_tracking[cur_model]["best_rmse"] = 100
    json.dump(model_tracking, open("model_tracker.json", "w"))
    try:
        shutil.rmtree(os.path.normpath(model_path))
    except:
        print("Unable to delete the old model information...\n\n")
        pass

if cur_model == "CNN[5533]_nodrop" or cur_model == "CNN[5533]_drop":
    conv_size = 5
else:
    conv_size = 3

####### GET THE TIME OF THE RUN FOR LOGGING #######
now = int(time.time())

####### PLACE TO SAVE THE MODEL AFTER TRAINING #######
# modelFn = os.path.normpath('models/tensorflow/steering_model.ckpt')
if not os.path.exists(os.path.normpath(model_path + "/checkpoints")):
    os.makedirs(model_path + "/checkpoints")

####### PLACE TO SAVE THE TENSORFLOW LOGS #######
logFn = os.path.normpath(model_path + "/logs")
if not os.path.exists(os.path.normpath(model_path + "/logs")):
    os.makedirs(model_path + "/logs")

####### PLACE TO SAVE THE OUTPUT FILES #######
ouptutDir = os.path.normpath('output')
if not os.path.exists('output'):
    os.makedirs('output')
outputFn = os.path.join(ouptutDir, str(now))

##############################################################################
##
## END: HOUSEKEEPING STUFF
##
##############################################################################




##############################################################################
##
## DATA INGEST
##
##############################################################################

print("\n\nReading in the data...\n\n")
trainImages = pickle.load(open('bigDataTrainImages.p', 'rb'))
valImages = pickle.load(open('bigDataTestImages.p', 'rb'))
testImages = pickle.load(open('challengeTestImages.p', 'rb'))
trainAngles = pickle.load(open('bigDataTrainAngles.p', 'rb'))
valAngles = pickle.load(open('bigDataTestAngles.p', 'rb'))
testAngles = pickle.load(open('challengeTestAngles.p', 'rb'))

##############################################################################
##
## END: DATA INGEST
##
##############################################################################




##############################################################################
##
## MODEL PARAMETERS
##
##############################################################################

batchSize = 10
n_iters = 50
n_classes = 1
learnRate = 0.0001


##############################################################################
##
## END: MODEL PARAMETERS
##
##############################################################################




##############################################################################
##
## TENSORFLOW HELPERS
##
##############################################################################

def weight_variable(shape):
    weights = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(weights, name='W')

def bias_variable(shape):
    bias = tf.ones(shape=shape)
    return tf.Variable(bias, name='B')

def conv_layer(input, width, height, channelsIn, channelsOut, name="conv", activation=tf.nn.leaky_relu):
    with tf.name_scope(name):
        w = weight_variable([width, height, channelsIn, channelsOut])
        b = bias_variable([channelsOut])
        conv = tf.nn.conv2d(input, w, strides=[1,1,1,1], padding='SAME')
        act = activation(conv + b)
        return tf.nn.max_pool(act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def fc_layer(input, channelsIn, channelsOut, name="fc", activation=tf.nn.leaky_relu):
    with tf.name_scope(name):
        w = weight_variable([channelsIn, channelsOut])
        b = bias_variable([channelsOut])
        return activation(tf.matmul(input, w) + b)

##############################################################################
##
## END: TENSORFLOW HELPERS
##
##############################################################################




##############################################################################
##
## COMPUTATION GRAPH
##
##############################################################################

with tf.name_scope('Data'):
    x = tf.placeholder(tf.float32, [None, 480, 640, 3], name='input_batch')
    y = tf.placeholder(tf.float32, [None], name='labels_batch')

with tf.name_scope('HyperParameters'):
    keepRate = tf.placeholder(tf.float32, name='keep_rate')

conv1 = conv_layer(x, conv_size, conv_size, 3, 64, 'conv1')
conv2 = conv_layer(conv1, conv_size, conv_size, 64, 32, 'conv2')
conv3 = conv_layer(conv2, 3, 3, 32, 16, 'conv3')
conv4 = conv_layer(conv3, 3, 3, 16, 8, 'conv4')

with tf.name_scope('flatten'):
    flat = tf.reshape(conv4, [-1, 40*30*8])

fc1 = fc_layer(flat, 40*30*8, 256, 'fc1')
fc1 = tf.nn.dropout(fc1, keepRate)
fc2 = fc_layer(fc1, 256, 64, 'fc2')
fc2 = tf.nn.dropout(fc2, keepRate)
out = fc_layer(fc2, 64, 1, 'out', activation=tf.identity)

##############################################################################
##
## END: COMPUTATION GRAPH
##
##############################################################################




##############################################################################
##
## EVALUATION METRICS
##
##############################################################################

with tf.name_scope('loss'):
    loss = tf.sqrt(tf.reduce_mean(tf.square(out - tf.reshape(y, [-1,1]))))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learnRate).minimize(loss)


##############################################################################
##
## END: EVALUATION METRICS
##
##############################################################################




##############################################################################
##
## TENSORBOARD SETUP
##
##############################################################################

saver = tf.train.Saver()
merged = tf.summary.merge_all()
### Start tensorboard ###
subprocess.Popen(["tensorboard", "--logdir", os.path.normpath("models/tensorflow")])
print("\n\n")

##############################################################################
##
## END: TENSORBOARD SETUP
##
##############################################################################




##############################################################################
##
## TENSORFLOW SESSION
##
##############################################################################

init_op = tf.global_variables_initializer()

print("Starting the tensorflow session...\n\n")
with tf.Session() as sess:

    print("Initializing the tensorflow variables...\n\n")
    sess.run(init_op)

    train_writer = tf.summary.FileWriter(logFn + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logFn + '/test', sess.graph)

    ##############################################################################
    ##
    ## TRAIN MODEL FROM SCRATCH
    ##
    ##############################################################################

    if args.action == 'train':
        print("Beginning the initial training of the model ", cur_model, "...\n\n")
        for i in range(n_iters):
            print("Beginning training epoch ", i, "\n\n")
            n_batches = int(trainImages.shape[0] / batchSize + 1)
            trainRMSE = 0
            for j in range(n_batches):
                batchImgs = []
                for imgID in trainImages[j*batchSize:(j+1)*batchSize]:
                    imgPath = os.path.normpath("Data/BigData/JPG/" + str(imgID) + ".jpg")
                    tempImg = misc.imread(imgPath)
                    batchImgs.append(tempImg)
                batchImgs = np.array(batchImgs)
                batchLabels = trainAngles[j*batchSize:(j+1)*batchSize]
                if (i*n_batches + j) % 100 == 0:
                    print("Batch %d out of %d completed..." % (j, n_batches))
                    summary, tempRMSE, _ = sess.run([merged, loss, train_step], feed_dict={x:batchImgs, y:batchLabels, keepRate:keep_rate})
                    trainRMSE = trainRMSE + tempRMSE
                    train_writer.add_summary(summary, i*n_batches + j)
                else:
                    tempRMSE, _ = sess.run([loss, train_step], feed_dict={x:batchImgs, y:batchLabels, keepRate:keep_rate})
                    trainRMSE = trainRMSE + tempRMSE

            trainRMSE = trainRMSE / n_batches
            rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='RMSE', simple_value=trainRMSE)])
            train_writer.add_summary(rmse_summary, i)

            ####### SAVE THE MODEL'S PROGRESS AFTER EACH EPOCH COMPLETES ######
            save_path = saver.save(sess, os.path.normpath(model_path + "/checkpoints/epoch" + str(i) + ".ckpt"))
            print("Model saved to %s at epoch %d" % (save_path, i))

            print("Epoch %d completed. Checking the validation effectiveness..." % i)
            n_batches = int(valImages.shape[0] / batchSize + 1)
            RMSE = 0
            for j in range(n_batches):
                batchImgs = []
                for imgID in valImages[j*batchSize:(j+1)*batchSize]:
                    imgPath = os.path.normpath("Data/BigData/JPG/" + str(imgID) + ".jpg")
                    tempImg = misc.imread(imgPath)
                    batchImgs.append(tempImg)
                batchImgs = np.array(batchImgs)
                batchLabels = valAngles[j*batchSize:(j+1)*batchSize]
                tempRMSE = sess.run([loss], feed_dict={x:batchImgs, y:batchLabels, keepRate:1.0})
                RMSE = RMSE + tempRMSE[0] #For Some reason tempRMSE is returned as a list
            RMSE = RMSE / n_batches
            rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='RMSE', simple_value=RMSE)])
            test_writer.add_summary(rmse_summary, i)
            print("Average RMSE after %d epochs: %f\n\n" % (i, RMSE))

            ####### UPDATE THE MODEL TRACKING INFO AFTER EACH EPOCH ########
            model_tracking[cur_model]["epoch"] = i
            model_tracking[cur_model]["train_rmse"] = trainRMSE
            model_tracking[cur_model]["val_rmse"] = RMSE
            if RMSE < model_tracking[cur_model]["best_rmse"]:
                model_tracking[cur_model]["best_rmse"] = RMSE
                model_tracking[cur_model]["best_epoch"] = i
            json.dump(model_tracking, open("model_tracker.json", "w"))

    ##############################################################################
    ##
    ## END: TRAIN MODEL FROM SCRATCH
    ##
    ##############################################################################




    ##############################################################################
    ##
    ## CONTINUE TRAINING
    ##
    ##############################################################################

    elif args.action == 'continue':
        print("Continuing the training of the model ", cur_model, "...\n\n")
        
        try:
            saver.restore(sess, os.path.normpath(model_path + "/checkpoints/epoch" + str(start_epoch) + ".ckpt"))
        except:
            print("Could not load the specified model...\n\n")
            sys.exit(1)

        for i in range(start_epoch, n_iters):
            print("Beginning training epoch ", i, "\n\n")
            n_batches = int(trainImages.shape[0] / batchSize + 1)
            trainRMSE = 0
            for j in range(n_batches):
                batchImgs = []
                for imgID in trainImages[j*batchSize:(j+1)*batchSize]:
                    imgPath = os.path.normpath("Data/BigData/JPG/" + str(imgID) + ".jpg")
                    tempImg = misc.imread(imgPath)
                    batchImgs.append(tempImg)
                batchImgs = np.array(batchImgs)
                batchLabels = trainAngles[j*batchSize:(j+1)*batchSize]
                if (i*n_batches + j) % 100 == 0:
                    print("Batch %d out of %d completed..." % (j, n_batches))
                    summary, tempRMSE, _ = sess.run([merged, loss, train_step], feed_dict={x:batchImgs, y:batchLabels, keepRate:keep_rate})
                    trainRMSE = trainRMSE + tempRMSE
                    train_writer.add_summary(summary, i*n_batches + j)
                else:
                    tempRMSE, _ = sess.run([loss, train_step], feed_dict={x:batchImgs, y:batchLabels, keepRate:keep_rate})
                    trainRMSE = trainRMSE + tempRMSE

            trainRMSE = trainRMSE / n_batches
            rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='RMSE', simple_value=trainRMSE)])
            train_writer.add_summary(rmse_summary, i)

            ####### SAVE THE MODEL'S PROGRESS AFTER EACH EPOCH COMPLETES ######
            save_path = saver.save(sess, os.path.normpath(model_path + "/checkpoints/epoch" + str(i) + ".ckpt"))
            print("Model saved to %s at epoch %d" % (save_path, i))

            print("Epoch %d completed. Checking the validation effectiveness..." % i)
            n_batches = int(valImages.shape[0] / batchSize + 1)
            RMSE = 0
            for j in range(n_batches):
                batchImgs = []
                for imgID in valImages[j*batchSize:(j+1)*batchSize]:
                    imgPath = os.path.normpath("Data/BigData/JPG/" + str(imgID) + ".jpg")
                    tempImg = misc.imread(imgPath)
                    batchImgs.append(tempImg)
                batchImgs = np.array(batchImgs)
                batchLabels = valAngles[j*batchSize:(j+1)*batchSize]
                tempRMSE = sess.run([loss], feed_dict={x:batchImgs, y:batchLabels, keepRate:1.0})
                RMSE = RMSE + tempRMSE[0] #For Some reason tempRMSE is returned as a list
            RMSE = RMSE / n_batches
            rmse_summary = tf.Summary(value=[tf.Summary.Value(tag='RMSE', simple_value=RMSE)])
            test_writer.add_summary(rmse_summary, i)
            print("Average RMSE after %d epochs: %f\n\n" % (i, RMSE))

            ####### UPDATE THE MODEL TRACKING INFO AFTER EACH EPOCH ########
            model_tracking[cur_model]["epoch"] = i
            model_tracking[cur_model]["train_rmse"] = trainRMSE
            model_tracking[cur_model]["val_rmse"] = RMSE
            if RMSE < model_tracking[cur_model]["best_rmse"]:
                model_tracking[cur_model]["best_rmse"] = RMSE
                model_tracking[cur_model]["best_epoch"] = i
            json.dump(model_tracking, open("model_tracker.json", "w"))

    ##############################################################################
    ##
    ## END: CONTINUE TRAINING
    ##
    ##############################################################################





    ##############################################################################
    ##
    ## TEST
    ##
    ##############################################################################

    elif args.action == 'test':

        try:
            saver.restore(sess, os.path.normpath(model_path + "/checkpoints/epoch" + str(model_tracking[cur_model]["best_epoch"]) + ".ckpt"))
        except:
            print("Unable to load model params from disk...\n\n")
            sys.exit(1)

        n_batches = int(testImages.shape[0] / batchSize + 1)
        RMSE = 0
        for j in range(n_batches):
            batchImgs = []
            for imgID in testImages[j*batchSize:(j+1)*batchSize]:
                imgPath = os.path.normpath("Data/BigData/TestImages/" + str(imgID) + ".jpg")
                tempImg = misc.imread(imgPath)
                batchImgs.append(tempImg)
            batchImgs = np.array(batchImgs)
            if len(batchImgs > 0):
                tempPred = sess.run([out], feed_dict={x: batchImgs, keepRate: 1.0})
            else:
                continue
            if j == 0:
                predictions = tempPred[0]
            else:
                predictions = np.concatenate((predictions, tempPred[0]))
            if j % 100 == 0:
                print("Batch %d predictions of %d batches completed..." % (j, n_batches))
        err = predictions - testAngles
        rms = np.sqrt(np.mean(err**2))
        print("RMSE on the test set: ", rms, "\n\n")

    ##############################################################################
    ##
    ## END: TEST
    ##
    ##############################################################################


##############################################################################
##
## END: TENSORFLOW SESSION
##
##############################################################################




