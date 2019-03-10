# Custom Mask-RCNN model on Real-Time Video Guide

Hey guys and welcome back so in this lecture, I am going to show you how to apply your trained custom Mask RCNN model from Supervisely, into your own python detector. This detector will be able to detect potholes in real time. 

The test video we shall be using is basically a combination of images from the test dataset, as we weren’t really able to find decent quality video online to test with. You can download the test video from the link attached to this video, or if you want test it with your own video. 

Now just note that the results you see here are not prefect as we trained our detector with only 200 annotated images. If you trained your model, it should perform better since we added in an additionally 200 labelled images. 
Okay so let’s pick up where we left off from the last lecture. 

## Download the Model
First we need to download the model, you go over here and click download.

INPUT_VIDEO - http://augmentedstartups.info/pothole-input-video

Place the extracted model in your python Mask RCNN folder.

## Adapting the Code
Firstly in the visualize_cv2.py we need to change our class names to match our application. Highlight these lines under the class_names list and let’s comment it out. 

*	Instead we will set our list to BG or background and Potholes 
*	We can then change our COCO_MODEL_PATH to model.h5 which is the downloaded model from Supervisely.

*		```NUM_CLASSES = num of classes + 1```

*		```IMAGE_MIN_DIM = 256``` (as per trained model parameter in supervisely).

*		```IMAGE_MAX_DIM=256```(as per trained model parameter in supervisely).

## Running the Custom Detector

In Anaconda prompt or command prompt and within our Mask RCNN directory, lets type in:

*	```python video_demo.py``` and then the name of the video which is ```input.avi``` and press enter. 

So it would be: ```python video_demo.py input.avi```

We’ll just wait for it to initiate before it executes the real time instance segmentation on the video.
Now as you can see are detecting some potholes. As mentioned earlier, that this model will not perform adequately for practical applications, however it shows the proof of concept. If we had more annotated images in our dataset (say over 2000 images) we can see some improvement.

Now if the detector is running a bit slow, we can increase the performance by adding this one line

```frame = cv2.resized(frame,(0,0),fx=0.5,fy=0.5)``` within the while loop. 

*	Run the detector

And there you have it. Share in the comments if you have implemented this on your own dataset and what your results were like. 
