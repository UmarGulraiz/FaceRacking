# FaceRacking
Our project is a human tracking system using face recognition. To reduce manpower and save time. Due to insufficient resources, we have just developed an automated system which marks attendance based on facial recognition on real time. By the end of a session our system generates a detailed report of attendance and sends it in the form of email to the owner. This product can be used in many aspects such as Provincial or national assembly attendance or in banks, etc. This product is almost fully human independent and is machine dependent and requires minimum human interaction. This project can be used in many ways. From a manager at building sites keeping track of workers to teachers in class room wasting class time in marking attendance, FaceRacking can be used to reduce manpower and to save time.

# Deployment/Installation Guide
We will provide complete installation. Also a USB will be provided which contains the interface.
4.0	User Manual

# Pre-requisites:

•	Cameras should be working.
•	There should be a training video.
•	Python should be installed.
•	Following libraries should be installed:
i.	Open CV
ii.	Numpy
iii.	Tensorflow
iv.	PIL
v.	Time
vi.	OS


Steps for user to start attendance module:

1-	Open CMD
2-	Move to your working directory


If Data set is not ready:

•	Move your video into the folder named “Data_Set”

•	Run file “VideoReader” on CMD using the command python VideoReader.py, this will give you all frames from the video in a folder named “unlabeled_dataset”.

•	Run the file “FacesFromIamges” on CMD using the command python FacesFromImges.py. This will extract all faces from the frames.

•	For the first use: the user/admin has to label the data themselves. Then paste the labialized data in a folder names “train_img”

If Data set is ready:
	
	Skip the above steps and continue from step 2
3-	Run file”data_preprocess” on CMD using the command python data_preprocess.py. (This will resize all the images into a folder named pre_img)

4-	Run the file “train_main” on CMD using the command python train_main.py. (This will train our model)

5-	Run the file “identify_face” on CMD using the command python identify_face.py. (This will train our model)

6-	Create a file in your directory named “contact.txt”. Store the owner’s name and Email in this file (A detailed report will be sent to this Email)
