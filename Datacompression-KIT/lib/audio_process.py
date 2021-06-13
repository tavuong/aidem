# audio_process.py
# Object oriented - 1D Audio Processing
# Datum : 13.04.2021
# Authors: Dr.-Ing. The Anh Vuong 
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import simpleaudio as sa
# Audio file Play
# https://realpython.com/playing-and-recording-sound-python/
# Test from_wave_file
# https://www2.cs.uic.edu/~i101/SoundFiles/

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
