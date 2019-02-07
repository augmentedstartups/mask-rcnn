# Mask RCNN Live video demo

To run Mask RCNN demo on video file we need to covert masked image in OpenCV format.

## Get all the files from Github

1.	First, we download script name “visualize_cv2.py” from GitHub. 

https://github.com/markjay4k/Mask-RCNN-series/blob/master/visualize_cv2.py

2.	Put this file in Mask-RCNN folder.

## Make changes to files

3.	This file uses other Mask RCNN library so we have to change in line no 6 and 7. Instead of ```import utils``` we will use ```from mrcnn import utils```.

4.	Also, coco library is inside ```sample/coco/``` so we have to go to that directory and after that we can import coco library.

5.	Another File we will write to read camera or video frame and pass that frame to Mask-RCNN model for segmentation

6.	In this file we will import cv2, sys and visualize_cv2 libraries.

7.	To get command line input from user we use ```argv```.

## Run Mask RCNN 

8.	Then, initialize Video capture function of OpenCV. Ensure your web-cam is connected.

9.	After that in while loop read camera or video frames and passes to the model.

10.	```display_instances``` function will apply mask on input image and return masked image.
