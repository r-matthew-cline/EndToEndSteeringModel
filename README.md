# EndToEndSteeringModel
End to end steering model for the Udacity self-driving car challenge #2. Uses a convolutional neural network to make steering decisions based on the camera input from the front facing camera on the car.

## Files
steering_model_keras.py - the main file for building the model, training and evaluation  
drive_keras.py - web app to interact with the udacity self-driving-car simulator  
data_manipulation.py - handles creating all of the pickle objects that store the image and label information  

### steering_model_keras.py
__Description:__ used to manage the network, the command line options give control  

__--model__ - the directory where the model checkpoints are stored  
__--action__ - train, continue, test, info, saliency  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_train_ - start the training of the model from scratch  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_continue_ - continue the training of the model from the initial epoch specified  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_test_ - evaluate the performance of the modelm on the test set  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_info_ - display the structure of the model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_saliency_ - examine an example saliency image to see where activations are happening  

__--init_epoch__ - the epoch to load  
__--batch__ - batch size to use  
__--iters__ - the number of epochs to train to  

### drive_keras.py
__Description:__ used to control the vehicle in the Udacity self-driving-car simulator  

__--model__ - the directory where the model checkpoints are stored  
__--epoch__ - the version of the model to load based on epoch  
__--hist__ - 1 to perform histogram equalization on the images, 0 for no equalization  
__--norm__ - 1 to perform normalization on the images, 0 for no normalization  

### data_manipulation.py
__Description:__ creates pickle objects to store a list of the image names in the training, validation, and test sets as well as their labels for easy loading. File structure should conform as follows:  

_./Data/Train/Images/_ - All training and validation images  
_./Data/Train/Labels/_ - Labels from the large training set  
_./Data/Train/interpolated.csv_ - labels from the smaller training set  
_./Data/Test/Images/_ - All test images from Ch2_001  
_./Data/Test/testLabels.csv_ - All test labels  

