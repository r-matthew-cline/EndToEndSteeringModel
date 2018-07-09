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

### Get the names of all the label files ###
i = 0
for file in os.listdir(os.path.normpath('Data/BigData/Labels')):
    if i < 1:
        df = pd.read_csv(os.path.join('Data/BigData/Labels', file), usecols=['timestamp', 'frame_id', 'angle'], index_col=False, dtype={'timestamp': object})
        i+=1
    else:
        df = pd.concat([df, pd.read_csv(os.path.join('Data/BigData/Labels',file), usecols=['timestamp', 'frame_id', 'angle'], index_col=False, dtype={'timestamp': object})])

df = df.loc[df['frame_id'] == 'center_camera']

testDF = pd.read_csv(os.path.normpath("Data/BigData/testLabels.csv"), usecols=['frame_id', 'steering_angle'], index_col=False, dtype={'frame_id': object})

# ### SHUFFLE THE DATA ###
# # print("Shuffling the data...\n\n")
# # shuffledData = shuffleData(np.array(data, copy=True))

### SPLIT THE DATA ###
print("Splitting the data into train and test...\n\n")
train, test = splitData(df)

### SPLIT THE IMAGE ID FROM THE STEERING ANGLE ###
print("Splitting the image id and angle...\n\n")
# trainImagesShuffled = trainShuffled[:,0]
# trainAnglesShuffled = trainShuffled[:,1]
# testImagesShuffled = testShuffled[:,0]
# testAnglesShuffled = testShuffled[:,1]
trainImages = np.array(train.iloc[:,0])
trainAngles = np.array(train.iloc[:,2])
testImages = np.array(test.iloc[:,0])
testAngles = np.array(test.iloc[:,2])
challengeTestImages = np.array(testDF.iloc[:,0])
challengeTestAngles = np.array(testDF.iloc[:,1])

### DUMP THE DATA TO PICKLE FILES FOR QUICK ACCESS ###
print("Dumping to the pickle files...\n\n")
# pickle.dump(trainImagesShuffled, open("trainImagesShuffled.p", "wb"))
# pickle.dump(trainAnglesShuffled, open("trainAnglesShuffled.p", "wb"))
# pickle.dump(testImagesShuffled, open("testImagesShuffled.p", "wb"))
# pickle.dump(testAnglesShuffled, open("testAnglesShuffled.p", "wb"))
pickle.dump(trainImages, open("bigDataTrainImages.p", "wb"))
pickle.dump(trainAngles, open("bigDataTrainAngles.p", "wb"))
pickle.dump(testImages, open("bigDataTestImages.p", "wb"))
pickle.dump(testAngles, open("bigDataTestAngles.p", "wb"))
pickle.dump(challengeTestImages, open("challengeTestImages.p", "wb"))
pickle.dump(challengeTestAngles, open("challengeTestAngles.p", "wb"))


