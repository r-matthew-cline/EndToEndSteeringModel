##############################################################################
##
## resize_images.py
##
## @author: Matthew Cline
## @version: 20180702
##
## Description: Simple script to ensure that all of the images in the 
## data directory are the appropriate size for the network.
##
##############################################################################

from PIL import Image
import os

directory = os.path.normpath('Data/AllImages')

print("Checking the size of all images in the data directory...")
for file_name in os.listdir(directory):
    image = Image.open(os.path.join(directory, file_name))
    x,y = image.size

    if x != 640 and y != 480:

        real_dim = (640, 480)
        print("Found an image of a different size. X=%d, Y=%d" % (x, y))
        # output = image.resize(real_dim, Image.ANTIALIAS)

        # output_file_name = os.path.join(directory, file_name)
        # output.save(output_file_name, "JPEG", quality=95)

print("Resizing of all images complete.")