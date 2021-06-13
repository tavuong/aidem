Download and install Anaconda from : https://docs.anaconda.com/anaconda/install/

open anaconda prompt and  navigate to the order Bild_kompression

cd path\Bild_kompression
(base) ...\Bild_kompression>conda create --name ENV_NAME python=3.6 cudatoolkit=10.0 cudnn
conda activate ENV_NAME
pip install tensorflow==2.0