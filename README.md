# droplet-control
Composite system for droplet control
This program is designed for automated control of droplet generation and consists of three main functional components: 1) Droplet recognition and calculation; 2) PID-driven automated droplet control; 3) Droplet frequency prediction and anomaly detection based on composite machine learning models.

The program comprises 1 main program, 5 subroutines, and 1 configuration file. It runs through the main program file "main_beta09.py". Proper configuration of the machine vision training files and frequency prediction model files is required. All necessary model files are located in the model folder.

Different machine learning models must be trained under different experimental conditions to ensure accuracy of frequency prediction. The "Data Collection" button in the main program allows for automatic collection of required data.

This program needs to be used in conjunction with pressure pump control software. For details, please contact the supplier Prinzen Biomedical Inc. (https://www.fluidiclab.com/).
