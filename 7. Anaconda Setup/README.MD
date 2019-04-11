# Installation Guide for Anaconda
Hey guys and welcome back. This lecture is going to be a quick lesson on how to install Anaconda for Mask RCNN. Lets get started. 
First we are going to go to Google type in Anaconda download and click on the first link.
## Installing Anaconda 

* Select Anaconda for windows and download Python 3.7
  * And then you can save it  anywhere
* Once it has downloaded you can open up the installer and follow the instructions.
  *	Everything can be left default except for advance options where we recommend that you Add Anaconda to my PATH Environmental variable 
  *	Click next and let it install		
* Now open Anaconda prompt or command from windows start menu.

## Creating the Environment 

Now type in this command :

 ```conda env create –f file_name.yml```.
 
which will create a new conda environment using .yml file.  This .yml file you will find in the github repo attached to this video. Ensure also that you are in the correct folder before we 
  *	Press enter to execute.
* Now to activate this new conda environment we execute the following command:

```conda activate mask_rcnn```
  *	You should see your new environment activate showing mask_rcnn in brackets.
 
## Installing the Dependencies

* Lastly we shall install all the other dependencies using the pip command.
  *	 ```pip install –r requirments.txt```
* To test if everything is working we can type in the command
  *	```Python demo.py```
You should get the response that is displayed over if everything was imported correctly. 


Okay that it is from me. In the next lecture I will show you how to setup and run Mask RCNN. 

