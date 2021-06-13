# Test_Recognistion with_Neuronal_Netzwork 
# Installation of environment
# Install for Windows
## MiniConda- Python  download from Miniconda https://docs.conda.io/en/latest/miniconda.html

##  Install python (miniconda)

## open "Anaconda Prompt" in Desktop

cd Drive:\...\app_Test_RecNN
(base) ...\app_Test_RecNN> conda env create -f environment_Recog.yml
(base) ...\app_Test_RecNN> conda activate Recog
(Recog) ...\app_AIServer>

## You have now a Enviroment with Tensorflow Bibbliothek for your Tensorflow Developmment with pc camera
## Testen command

(Recog) app_Test_RecNN>python TFLite_detection_webcam.py --modeldir=Sample_Modell


# Installation for Raspberry with Camera
cd /home/pi
sudo apt-get update
sudo apt-get upgrade
## install virtual environment
python3 -m venv plask-cv-env
cd  plask-cv-env
source plask-cv-env/bin/activate

##  open cv 
## use get_pi_requirement.sh
(credit: git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git)

bash get_pi_requirements.sh
pip3 install flask
source plask-cv-env/bin/activate

(plask-cv-env) Drive:\...\app_AIServer>

--- You have now a Enviroment with Tensorflow, opencv  and Flask for your Developmment with RPI and camera

# Installation Tensorflow Python-Enviroment 

cd Drive:\...\app_AIServer

(base) Drive:\...\app_AIServer> conda env create -f environment_Tensorflow.yml

(base) Drive:\...\app_AIServer> conda activate Tensorflow

(Tensorflow) Drive:\...\app_AIServer>

--- You have now a Enviroment with Tensorflow Bibbliothek for your Tensorflow Developmment


Have fun!

Vuong-AIDEM Team

----
### NOTE: 
Licence is attached in each app
Source code could be download from https://sourceforge.net/p/vuong-aidem
