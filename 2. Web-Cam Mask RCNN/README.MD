# Mask RCNN Live Video Demo

Hey guys and welcome back, In the last lecture we spent setting up all the required libraries and dependencies to get mask rcnn in to process a single image. In this lecture I will show you how to set up real-time mask rcnn using either a webcam or process recorded video.

Okay so lets get started on real time image segmentation on Windows 10.


To run Mask RCNN demo on video file we need to covert masked image in OpenCV format.

## Get all the Files from Github

To run Mask RCNN demo on video file we need to convert the masked image code to OpenCV format.

1.	First, we download script name “visualize_cv2.py” from GitHub. 

https://github.com/markjay4k/Mask-RCNN-series/blob/master/visualize_cv2.py

2.	Put this file in Mask-RCNN folder.

## Make Changes to Files or Download the Code

3.	This file uses other Mask RCNN library so we have to change in line no 6 and 7. Instead of ```import utils``` we will use ```from mrcnn import utils```.

4.	Also, coco library is inside ```sample/coco/``` so we have to go to that directory and after that we can import coco library.

5.	Another File we will write to read camera or video frame and pass that frame to Mask-RCNN model for segmentation

6.	In this file we will import cv2, sys and visualize_cv2 libraries.

7.	To get command line input from user we use ```argv```.

8.	Then, initialize Video capture function of OpenCV. Ensure your web-cam is connected.

9.	After that in while loop read camera or video frames and passes to the model.

10.	```display_instances``` function will apply mask on input image and return masked image.

If you rather not follow along with the code you can copy these two python files at this location:
https://github.com/reigngt09/mask-rcnn/tree/master/2.%20Web-Cam%20Mask%20RCNN

## Run Mask RCNN 
To run the code, all you have to do is ```cd``` into the Mask RCNN folder and run the following command:

11.	```python video_demo.py 0```  for webcam or

12.	```python video_demo.py video.mp4```  to run on a video (replace video.mp4 with the actual video file name)

