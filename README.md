# EndToEndSteeringModel
End to end steering model for the Udacity self-driving car challenge #2. Uses a convolutional neural network to make steering decisions based on the camera input from the front facing camera on the car.

## Files
steering_model_keras.py - the main file for building the model, training and evaluation  
drive_keras.py - web app to interact with the udacity self-driving-car simulator  
data_manipulation.py - handles creating all of the pickle objects that store the image and label information  

### steering_model_keras.py
Description: used to manage the network, the command line options give control  

--model - the directory where the model checkpoints are stored  
--action - train, continue, test, info, saliency  

   train - start the training of the model from scratch  
   continue - continue the training of the model from the initial epoch specified  
   test - evaluate the performance of the modelm on the test set  
   info - display the structure of the model  
   saliency - examine an example saliency image to see where activations are happening  

--init_epoch - the epoch to load  
--batch - batch size to use  
--iters - the number of epochs to train to  

### drive_keras.py
Description: used to control the vehicle in the Udacity self-driving-car simulator  

--model - the directory where the model checkpoints are stored  
--epoch - the version of the model to load based on epoch  
--hist - 1 to perform histogram equalization on the images, 0 for no equalization  
--norm - 1 to perform normalization on the images, 0 for no normalization  

### data_manipulation.py
Description: creates pickle objects to store a list of the image names in the training, validation, and test sets as well as their labels for easy loading. File structure should conform as follows:  

./Data/Train/Images/ - All training and validation images  
./Data/Train/Labels/ - Labels from the large training set  
./Data/Train/interpolated.csv - labels from the smaller training set  
./Data/Test/Images/ - All test images from Ch2_001  
./Data/Test/testLabels.csv - All test labels  

