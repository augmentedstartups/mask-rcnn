# Guide to test custom Mask-RCNN model on live video
	We first add in some lines in visualize_cv2.py change NUM_CLASSES,class_names as per our dataset and add image dimentations. 
	```NUM_CLASSES = num of classes + 1```
	```IMAGE_MIN_DIM = 256``` (as per trained model parameter in supervisely).
	```IMAGE_MAX_DIM=256```(as per trained model parameter in supervisely).
	
	class_names = list of object in string
	
	Change model name in COCO_ MODEL_PATH.
	Now in video_demo.py
	Add  ```frame = cv2.resized(frame,(0,0),fx=0.5,fy=0.5)``` within the while loop.
