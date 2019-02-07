# Installation Guide for Mask-RCNN
Hey guys and welcome back, in this lecture im going  to show you how to install all the dependencies as well as all the tools required to get Mask RCNN up and running. This lecture we will show you how to process a single image and the next lecture I will show you hot to get it working on video. Okay so  lets get right into it. 

# Github Repo Download
1.	 Download Mask-RCNN repository for GitHub - https://github.com/matterport/Mask_RCNN

```Use git clone https://github.com/matterport/Mask_RCNN.git```

2.	Go to: Mask_RCNN folder

3.	This repository has "requirements.txt" file which contains dependencies of Mask-RCNN. I will have my full requirements file on my github page in the link below, which will help save you time.

4.	Also download pretrained model from here: mask_rcnn_coco - https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

5.	Move downloaded model to MASK_RCNN folder

6.	Remove TensorFlow from that text file because we want to use TensorFlow-GPU.

# Download and Install CUDA
1.	TensorFlow GPU requires CUDA. So, we will install CUDA 9.0 for Windows 10. With a help of this link you can download CUDA 9.0 for windows. - https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_win10-exe

2.	While CUDA is downloading we will install other dependency.
  
# Installing the Dependencies
1.	If System has installed Anaconda, then activate anaconda’s environment to install python dependencies otherwise user can directly install dependencies using “pip”.

2.	Command: ```pip install –r requiments.txt``` to install dependencies

3.	Mask-RCNN also requires cocoapi. To install cocoapi go to - https://github.com/philferriere/cocoapi

4.	Open: "cocoapi-master" folder

5.	Go to: "PythonAPI" folder

6.	Run in CMD: ```python setup.py build_ext install```.

7.	Now, Install CUDA using Executable file.

8.	After that, Install TensorFlow GPU using: ```pip install tensorflow-gpu```.


# Running Mask RCNN demo

1.	Run command: ```jupyter-notebook```.

2.	Navigate to samples folder open demo.jpynb.

3.	Select download as python option from File menu. 

4.	Comment line no.32. ```get_ipython().run_line_magic(‘matplotlib’,’inline’)```

5.	Run python file. ```python demo.py```
 
# Possible Errors and solutions
•	Error while running command: ```pip install –r requirments.txt```

•	Go to: Windows binaries - https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely

•	Download Shapely according to python version.

•	Run CMD: ```pip install downloaded file.whl```

•	Re run command: ```pip install –r requiments.txt```

•	Cython error while installing cocoapi.

•	If system has installed Cython and reinstall Cython.

•	Command to uninstall Cython: ```conda uninstall cython```

•	Command to install Cython: ```conda install cython```
