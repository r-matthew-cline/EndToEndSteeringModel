##############################################################################
##
## data_manipulation.py
##
## @author: Matthew Cline
## @version: 20180529
##
## Description: Preperation of the training and test data for the end to 
## end steering model. The images for training of the model are center
## cropped jpg images. The two training sets are combined and shuffled. The 
## test set is kept sparate for evaluation purposes.
##
##############################################################################

import pandas as pd
import numpy as np
import pickle
import os
import sys

np.set_printoptions(suppress=True)

def splitData(data, trainingSplit=0.8):
    training, test = np.split(data, [int(data.shape[0] * trainingSplit)])
    return training, test

def shuffleData(data):
    data = data.reindex(np.random.permutation(data.index))
    data = data.reset_index(drop=True)
    return data

### Get the names of all the label files ###
i = 0
for file in os.listdir(os.path.normpath('Data/Train/Labels')):
    if i < 1:
        df = pd.read_csv(os.path.join('Data/Train/Labels', file), usecols=['timestamp', 'frame_id', 'angle'], index_col=False, dtype={'timestamp': object})
        i+=1
    else:
        df = pd.concat([df, pd.read_csv(os.path.join('Data/Train/Labels',file), usecols=['timestamp', 'frame_id', 'angle'], index_col=False, dtype={'timestamp': object})])

df = df.loc[df['frame_id'] == 'center_camera']
otherDf = pd.read_csv(os.path.normpath("Data/Train/interpolated.csv"), usecols=['timestamp', 'frame_id', 'angle'], index_col=False, dtype={'timestamp': object})
otherDf = otherDf.loc[otherDf['frame_id'] == 'center_camera']
df = pd.concat([df, otherDf], axis=0)
testDF = pd.read_csv(os.path.normpath("Data/Test/testLabels.csv"), usecols=['frame_id', 'steering_angle'], index_col=False, dtype={'frame_id': object})
curvesDf = df.copy()
curvesDf = curvesDf[abs(curvesDf['angle']) > 0.1]
print("Curves Samples = ", curvesDf.shape[0])

### SHUFFLE THE DATA ###
print("Shuffling the data...\n\n")
shuffledData = df.copy()
shuffledData = shuffledData.sample(frac=1).reset_index(drop=True)
shuffledCurves = curvesDf.sample(frac=1).reset_index(drop=True)


### SPLIT THE DATA ###
print("Splitting the data into train and test...\n\n")
train, test = splitData(df)
trainShuffled, testShuffled = splitData(shuffledData)
trainCurves, testCurves = splitData(shuffledCurves)


### SPLIT THE IMAGE ID FROM THE STEERING ANGLE ###
print("Splitting the image id and angle...\n\n")
trainCurvesImages = np.array(trainCurves.iloc[:,0])
trainCurvesAngles = np.array(trainCurves.iloc[:,2])
valCurvesImages = np.array(testCurves.iloc[:,0])
valCurvesAngles = np.array(testCurves.iloc[:,2])
trainImagesShuffled = np.array(trainShuffled.iloc[:,0])
trainAnglesShuffled = np.array(trainShuffled.iloc[:,2])
valImagesShuffled = np.array(testShuffled.iloc[:,0])
valAnglesShuffled = np.array(testShuffled.iloc[:,2])
trainImages = np.array(train.iloc[:,0])
trainAngles = np.array(train.iloc[:,2])
valImages = np.array(test.iloc[:,0])
valAngles = np.array(test.iloc[:,2])
testImages = np.array(testDF.iloc[:,0])
testAngles = np.array(testDF.iloc[:,1])

### DUMP THE DATA TO PICKLE FILES FOR QUICK ACCESS ###
print("Dumping to the pickle files...\n\n")
pickle.dump(trainImagesShuffled, open("trainImagesShuffled.p", "wb"))
pickle.dump(trainAnglesShuffled, open("trainAnglesShuffled.p", "wb"))
pickle.dump(valImagesShuffled, open("valImagesShuffled.p", "wb"))
pickle.dump(valAnglesShuffled, open("valAnglesShuffled.p", "wb"))
pickle.dump(trainImages, open("trainImages.p", "wb"))
pickle.dump(trainAngles, open("trainAngles.p", "wb"))
pickle.dump(valImages, open("valImages.p", "wb"))
pickle.dump(valAngles, open("valAngles.p", "wb"))
pickle.dump(testImages, open("testImages.p", "wb"))
pickle.dump(testAngles, open("testAngles.p", "wb"))
pickle.dump(trainCurvesImages, open("trainCurvesImages.p", "wb"))
pickle.dump(trainCurvesAngles, open("trainCurvesAngles.p", "wb"))
pickle.dump(valCurvesImages, open("valCurvesImages.p", "wb"))
pickle.dump(valCurvesAngles, open("valCurvesAngles.p", "wb"))


