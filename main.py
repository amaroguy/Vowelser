import signal
import sys
import numpy as np
import pygame
from visualize import FormantTracker
from formantstream import FormantStream
from voicestream import VoiceStream
from smoothers.mediansmoother import MedianStreamer

#pyaudio setup
#We'll be taking in about .1 seconds of data every time we do LPC w/2048 samples
#because of the 2048 / sampling rate
SAMPLING_RATE = 14000
BUFFER_SIZE = 1024 ## of 2 byte samples

#44100
#1024
#order 40
#(200, 1000)     
#(800 2400)                             
                                        #x dir<>      f2 y dir ^v
# visualizer = FormantTracker(800, 800, (200,800), (800,2100))

#x is f2, y is f1
visualizer = FormantTracker(800, 800, (800,2400), (200,800))

#https://en.wikipedia.org/wiki/Formant#/media/File:Average_vowel_formants_F1_F2.png
#Median
# /i/ - f1: 200
# /a/ - f1: 700

# .25 for f2
# median for f1?

voicestream = VoiceStream(SAMPLING_RATE, BUFFER_SIZE)

f1_smoother = MedianStreamer(60)
f2_smoother = MedianStreamer(60)

formantstream = FormantStream(voicestream, f1_smoother=f1_smoother, f2_smoother=f2_smoother)

f1_history = []
f2_history = []
# f3_history = []

#handle ctrl c
def handle_exit(arg1, arg2):
    print("Exiting!")

    f1_stats = np.quantile(np.array(f1_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f2_stats = np.quantile(np.array(f2_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    # f3_stats = np.quantile(np.array(f3_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])

    print(f"{f1_stats}")    
    print(f"{f2_stats}")
    # print(f"{f3_stats}")

    voicestream.destroy()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

sample = -1
alpha = 0.7




while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            handle_exit(1,2)

    f1, f2 = formantstream.get_formants(4)
    f1_history.append(f1) 
    f2_history.append(f2)
    
    #/i/ tl, deed
    #(234, 2450)
    #/a/ bl, dad
    #(780, 1650) 
    #/ɑ/ br
    #(720, 1250)
    #/u/ dud
    #(300, 900)

    #round 2
    #/i/ (200, 2100)
    #/a/ (700, 1600)
    #/ɑ/ (620, 1100)
    #/u/ (250, 800)
    print(f"f1: {int(f1)} f2: {int(f2)}, f3: {"NA"} f4: {"NA"}")


    visualizer.draw_formant(f2, f1)

    
