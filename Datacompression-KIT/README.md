# This is Develop kit for Datacompression of  Images
# Usage:
Environment need
## open CV2

pip install cv2
pip install mumpy
# für differences 

pip install panda
pip install workbook
pip install openpyxl

# for audio
pip install wave, simpleaudio

# Start Programm
$ python codecKIT-v1.py

old version:
$ python codecKIT-vuong.py

# Kit Stucture
- config.py: define for input / Output files
- codecKIT : frame work dashboard 
- lib :
    - block_process2 : block read, pocess and reconst
    - codec_dct      : demo  plug-in lib to block_procees: dvt_2, idct_2, lowpass_2d

# codecKIT-vuong.py
# In Git: Example  for datacompression by 64*64 Image, block: 16*16, lowpass_Filter 4*4
# In Git for Entwickler: dont push your processed Data to GIT
input file is defined in config.py 
output files are definde in config.py
python codecKIT-vuong.py

Input Block-length --> numberBlock
Input 2D Lowpass length ---> nbit (Lowpass-example : Lowpass nbit * nbit of spectrals, rest set to o)
blocks: ouput-folder with png-files for blocks

modus P =Process        : Blocks generate, DCT and Idct to reconstruction:
                            - blocks/ images blocks (*.png)
                            - spect/  spect- blocks (*.png)
                            - filter/ filter- blocks (*.png)
                            - recon/  recon- blocks (*.png)
                            img_out / movieori.gif file of image blocks
                                      moviespect.gif file of spect blocks
                                      moviefilter.gif file of filter blocks
                                      movierecont.gif file of recont blocks
                                      spectNN.jpg  spec-blocks in ful-images
                                      filterNN.jpg  filter of spec-blocks in ful-images
                                      reconNN.jpg reconstruction images

modus G= Blocks         : Blocks generate
                            - blocks/ images blocks (*.png)
                            img_out / movieori.gif file of blocks

modus M= Mini Block in Image   : Blocks in big Image generate
                                - blocks/ images blocks (*.png) but mappinfg in big images
                                img_out / movieori-map.gif file of blocks

# Project : Datacompression-KIT

Initiator: Dr.-Ing. The Anh Vuong (admin) 
Developer: Dr.-Ing. The Anh Vuong, Tim Orius
License: MIT