import signal
import sys
import pyaudio
import numpy as np
import librosa
import pygame
from visualize import FormantTracker
import statistics
from formantstream import FormantStream
from voicestream import VoiceStream

#pyaudio setup
#We'll be taking in about .1 seconds of data every time we do LPC w/2048 samples
#because of the 2048 / sampling rate
SAMPLING_RATE = 44100
BUFFER_SIZE = 1024 ## of 2 byte samples
visualizer = FormantTracker(800, 800, (600, 2400), (200,1000))

#https://en.wikipedia.org/wiki/Formant#/media/File:Average_vowel_formants_F1_F2.png
#Median
# /i/ - f1: 200
# /a/ - f1: 700

# .25 for f2
# median for f1?

voicestream = VoiceStream(SAMPLING_RATE, BUFFER_SIZE)
formantstream = FormantStream(voicestream, None)

f1_history = []
f2_history = []
f3_history = []

#handle ctrl c
def handle_exit(arg1, arg2):
    print("Exiting!")

    f1_stats = np.quantile(np.array(f1_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f2_stats = np.quantile(np.array(f2_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f3_stats = np.quantile(np.array(f3_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])

    print(f"{f1_stats}")    
    print(f"{f2_stats}")
    print(f"{f3_stats}")

    voicestream.destroy()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

sample = -1
alpha = 0.9




while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            handle_exit(1,2)

    formants = formantstream.get_formants(4)
    f1, f2, f3, f4 = formants[:4]
    f1_history.append(f1) 
    f2_history.append(f2)
    f3_history.append(f3)
    print(f"f1: {int(f1)} f2: {int(f2)}, f3: {int(f3)} f4: {f4}")


    visualizer.draw_formant(f3, f2)

    
