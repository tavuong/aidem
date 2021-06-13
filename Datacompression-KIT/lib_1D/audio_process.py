# audio_process.py
# Object oriented - 1D Audio Processing
# Datum : 13.04.2021
# Authors: Dr.-Ing. The Anh Vuong 
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import simpleaudio as sa
import lib_1D.Process_1D as P1D
from scipy.io import wavfile as wav
from scipy.fftpack import fft

"""
Audio file Play
https://realpython.com/playing-and-recording-sound-python/

Test from_wave_file
https://www2.cs.uic.edu/~i101/SoundFiles/

simpleaudio tuturial
https://simpleaudio.readthedocs.io/en/latest/tutorial.html
"""
def play(filename):
   # filename = './Musik/swift.wav'
   wave_obj = sa.WaveObject.from_wave_file(filename)
   play_obj = wave_obj.play()
   play_obj.wait_done()  # Wait until sound has finished playing
   return (True)

# Audio File plot
def plot(filename):
   spf = wave.open(filename, "r")
   p= spf.getparams()
   print(p)

   # Extract Raw Audio from Wav File
   signal = spf.readframes(-1)
   # Info
   signal = np.fromstring(signal, "Int16")
   # If Stereo
   if spf.getnchannels() == 2:
      print("Just mono files")
      sys.exit(0)
   
   plt.figure(1)
   plt.title("Signal Wave..." + filename)
   plt.plot(signal)
   plt.show()
# FFT
# 
#   rate, data = wav.read(filename)
   samplerate, data = wav.read(filename)
   print("number of samples ="+ str(samplerate))
#   chanel = data.shape[1]
#   print("number of channels =" + str(chanel))
   length = data.shape[0] / samplerate
   print("length = " + str(length) + " s")

   fft_out = fft(data)
   plt.plot(data, np.abs(fft_out))
   plt.show()
   