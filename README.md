# American Sign Language (ASL) to Text Application

Senior Project for San Jose State University's B.S. Computer Engineering Program
Graduating Class of 2018

## Description 
This project uses PYQT GUI to detect American Sign Language hand gestures through histogram recognition and outputs that into an output text box. 

## Preresiquites and Installation Guide
First you would need install these dependency through a terminal

1. Python 3.6.0+
2. Keras 2.2.4
3. Tensorflow 1.12.0+
4. OpenCV 3.4.0
5. PYQT5
6. Git

## How to Install Preresiquites
1. Open up Startup and click "Run"
2. Type "cmd" and press enter
3. select a directory (Documents) in which you want to have the project located
4. In your chosen directory, type the following command:
	'''
	git clone https://github.com/brentjsimon/ASL-to-Text-App.git
	'''
5. Once you get the project onto your directory, get into the branch
	'''
	git checkout development
	'''
6. Next download the requirements needed to run the program 
	If you are running on a GPU, run the following:
	'''
	pip install -r requirements_gpu.txt
	'''
	If you are running on a CPU, run the following:
	'''
	pip install -r requirements_cpu.txt
	'''
## How to run the program
1. Go to terminal and change your diretory to the project location in your PC
2. Once you are in the project directory, type in 
	'''
	ls
	'''
	or
	'''
	dir
	'''
	to see the current files in the folder
3. When you see the python file called "UI_ASL_App.py", run the following command to run the program
	'''
	python UI_ASL_App.py
	'''
4. You should then be prompted by a GUI window that will access your camera
5. Place your hand in front of the green square and then click the button "Capture Skin Color"
	This will allow you the camera to capture an outline of your hand.
	Keep on clicking the button until you see a clear white space histogram of your hand
6. Once you are satified with your captured white hand, select the start Translation bubble 
7. From here, you may gesture any American Sign Language letter, number or gesture. 


## Development Tools Used
1. PYQT5 Tools 
2. PGAdmin3+

## Hardware 
- Front Facing 2D Camera or Webcam


## Team

- Brent Simon 
- Kenny Leong 
- Danielle Maak 
- Andie Sanchez 
- Melissa Lauzon