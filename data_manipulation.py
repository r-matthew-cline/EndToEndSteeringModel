##############################################################################
##
## data_manipulation.py
##
## @author: Matthew Cline
## @version: 20180529
##
## Description: Preperation of the training and test data for the end to 
## end steering model. The images for training of the model are center
## cropped jpg images. The actual images used for evaluation are stored
## as ROS bag files. TO BE CONTINUED...
##
##############################################################################

import pandas as pd
import numpy as np
import pickle
import os

np.set_printoptions(suppress=True)

def splitData(data, trainingSplit=0.8):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    return data

print("Reading in the training data...\n\n")
labelsFn1 = os.path.normpath('Data/Ch2_001/final_example.csv')
labelsFn2 = os.path.normpath('Data/Ch2_002/output/steering.csv')
cameraMapFn = os.path.normpath('Data/Ch2_002/output/camera.csv')

### Clean data from part one of the contest ###
data1 = pd.read_csv(labelsFn1, dtype={'frame_id': object}) 

### Steering angle data from part 2 of the contest. The timestamps do not match the images. ###
data2 = pd.read_csv(labelsFn2, usecols=['timestamp', 'angle'], index_col = False, dtype={'timestamp': object}) 

### The image timestamps from part 2 of the contest. Do not match the steering angle timestamps. ###
cameraMap = pd.read_csv(cameraMapFn, dtype={'timestamp': object})

### Grab the timestamps for only the center camera images ###
centerImgs = cameraMap.loc[cameraMap['frame_id'] == 'center_camera']

### Split the image timestamps into an iterable list ###
includedTimestamps = centerImgs.loc[:,'timestamp'].tolist()

### Iterate through the image timestamps to correlate with the steering angel timestamps. ###
### All of the steering angles collected between the previous image capture and the next  ###
### image capture are extracted into a temporary data frame. The mode of the steering     ###
### angles is used to avoid having extreme changes in steering angle between frames       ###
### having large impacts on the average.                                                  ###
print("Beginning correlation detection between image timestamps and steering angle timestamps...\n\n")
for i in range(len(includedTimestamps)):

    if i == 0:
        angle = data2.loc[(data2['timestamp'] > includedTimestamps[i]) & (data2['timestamp'] < includedTimestamps[i+1])].mode().loc[0,'angle']
    elif i == len(includedTimestamps) - 1:
        angle = data2.loc[(data2['timestamp'] > includedTimestamps[i-1]) & (data2['timestamp'] < includedTimestamps[i])].mode().loc[0,'angle']
    else:
        angle = data2.loc[(data2['timestamp'] > includedTimestamps[i-1]) & (data2['timestamp'] < includedTimestamps[i+1])].mode().loc[0,'angle']

    data1.loc[len(data1)] = [includedTimestamps[i], angle]

    if (i+1) % 1000 == 0:
        print("Completed %d out of %d iterations..." % (i+1, len(includedTimestamps)))   

print(data1)

### SHUFFLE THE DATA ###
# print("Shuffling the data...\n\n")
# shuffledData = shuffleData(np.array(data, copy=True))

### SPLIT THE DATA ###
print("Splitting the data into train and test...\n\n")
# trainShuffled, testShuffled = splitData(shuffledData)
train, test = splitData(data1)

print(train)

### SPLIT THE IMAGE ID FROM THE STEERING ANGLE ###
print("Splitting the image id and angle...\n\n")
# trainImagesShuffled = trainShuffled[:,0]
# trainAnglesShuffled = trainShuffled[:,1]
# testImagesShuffled = testShuffled[:,0]
# testAnglesShuffled = testShuffled[:,1]
trainImages = np.array(train.iloc[:,0])
trainAngles = np.array(train.iloc[:,1])
testImages = np.array(test.iloc[:,0])
testAngles = np.array(test.iloc[:,1])

### DUMP THE DATA TO PICKLE FILES FOR QUICK ACCESS ###
print("Dumping to the pickle files...\n\n")
# pickle.dump(trainImagesShuffled, open("trainImagesShuffled.p", "wb"))
# pickle.dump(trainAnglesShuffled, open("trainAnglesShuffled.p", "wb"))
# pickle.dump(testImagesShuffled, open("testImagesShuffled.p", "wb"))
# pickle.dump(testAnglesShuffled, open("testAnglesShuffled.p", "wb"))
pickle.dump(trainImages, open("trainImages.p", "wb"))
pickle.dump(trainAngles, open("trainAngles.p", "wb"))
pickle.dump(testImages, open("testImages.p", "wb"))
pickle.dump(testAngles, open("testAngles.p", "wb"))


