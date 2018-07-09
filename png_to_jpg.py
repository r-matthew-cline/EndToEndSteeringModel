##############################################################################
##
## png_to_jpg.py
##
## @author: Matthew Cline
## @version: 20180707
##
## Description: Simple script to convert all of the png images in a folder 
## to jpg images and save them in another folder.
##
##############################################################################

import os
import sys
import scipy.misc as misc

if len(sys.argv) < 3:
    print("Enter an input directory as the first argument and an output directory as the second argument.")
    sys.exit(1)

inputDir = sys.argv[1]
outputDir = sys.argv[2]

if not os.path.exists(outputDir):
    os.makedirs(outputDir)

for file in os.listdir(inputDir):
    img = misc.imread(os.path.join(inputDir, file))
    misc.imsave(os.path.join(outputDir, file[:-3] + 'jpg'), img)
