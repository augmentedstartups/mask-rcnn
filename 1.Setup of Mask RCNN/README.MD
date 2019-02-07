# Installation Guide for Mask-RCNN
Hey guys and welcome back, in this lecture im going  to show you how to install all the dependencies as well as all the tools required to get Mask RCNN up and running. This lecture we will show you how to process a single image and the next lecture I will show you hot to get it working on video. Okay so  lets get right into it. 

# Github Repo Download
•	 Download Mask-RCNN repository for GitHub - https://github.com/matterport/Mask_RCNN

```Use git clone https://github.com/matterport/Mask_RCNN.git```

•	Go to: Mask_RCNN folder

•	This repository has "requirements.txt" file which contains dependencies of Mask-RCNN. I will have my full requirements file on my github page in the link below, which will help save you time.

•	Also download pretrained model from here: mask_rcnn_coco

•	Move downloaded model to MASK_RCNN folder

•	Remove TensorFlow from that text file because we want to use TensorFlow-GPU.

# Download and Install CUDA
•	TensorFlow GPU requires CUDA. So, we will install CUDA 9.0 for Windows 10. With a help of this link you can download CUDA 9.0 for windows.

https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_win10-exe

•	While CUDA is downloading we will install other dependency.
  
# Installing the Dependencies
•	If System has installed Anaconda, then activate anaconda’s environment to install python dependencies otherwise user can directly install dependencies using “pip”.

•	Command: ```pip install –r requiments.txt``` to install dependencies

•	Mask-RCNN also requires cocoapi. To install cocoapi go to: https://github.com/philferriere/cocoapi

•	Open: "cocoapi-master" folder

•	Go to: "PythonAPI" folder

•	Run in CMD: ```python setup.py build_ext install```.

•	Now, Install CUDA using Executable file.

•	After that, Install TensorFlow GPU using: ```pip install tensorflow-gpu```.


# Running Mask RCNN demo

•	Run command: ```jupyter-notebook```.

•	Navigate to samples folder open demo.jpynb.

•	Select download as python option from File menu. 

•	Comment line no.32. ```get_ipython().run_line_magic(‘matplotlib’,’inline’)```

•	Run python file. ```python demo.py```
 
# Possible Errors and solutions
•	Error while running command: ```pip install –r requirments.txt```

•	Go to: Windows binaries [Link]

•	Download Shapely according to python version.

•	Run CMD: ```pip install downloaded file.whl```

•	Re run command: ```pip install –r requiments.txt```

•	Cython error while installing cocoapi.

•	If system has installed cython and reinstall cython.

•	Command to uninstall Cython: ```conda uninstall cython```

•	Command to install Cython: ```conda install cython```
