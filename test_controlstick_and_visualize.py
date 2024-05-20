import signal
import sys
import pyaudio
import numpy as np
import librosa
import pygame
from visualize import FormantTracker
import statistics
from collections import deque

# This file visualizes the control stick/formant tracking and
# is also used to calibrate the control stick. I used this script for
# example to: 
#
# 1. start going /i/ before running it, and starting the script
# 2. keep going /i/ for ~5 seconds
# 3. halting the script and seeing the average f1/f2 that gets printed out
# 
# I rounded each of the recorded values DOWN to the nearest 100, and expanded the "radius"
# by about 100. 
# 
# I double checked the formants aligned with what Praat measured for the formants as well
# if they did not align with what Praat said, I fiddled with the number of LPC coefficients
# or with the BUFFER_SIZE for how much information was given to perform LPC.



#pyaudio setup
#We'll be taking in about .1 seconds of data every time we do LPC w/2048 samples
#because of the 2048 / sampling rate
SAMPLING_RATE = 44100
BUFFER_SIZE = 4096 ## of 2 byte samples


#/i/ -> (200, 2500)
#/a/ -> (480, 1600)
#/u/ -> (290, 1400)
#/aw/ -> (300, 1000)

#/i/ -> (200, 2700)
#/a/ -> (650, 1600)
#/u/ -> (220, 550)
#/aw/ -> (200, 800)

F2_RANGE = (500, 2800)
F1_RANGE = (100, 800)
visualizer = FormantTracker(800, 800, F2_RANGE, F1_RANGE)
f1_smoother = deque(maxlen=20)
f2_smoother = deque(maxlen=20)



#https://en.wikipedia.org/wiki/Formant#/media/File:Average_vowel_formants_F1_F2.png

pyaud = pyaudio.PyAudio()
audio_stream = pyaud.open(
    rate= SAMPLING_RATE, 
    channels=1,
    format=pyaud.get_format_from_width(2, True), #bit depth!
    input=True,
    output=False,
    input_device_index=0,
    frames_per_buffer=BUFFER_SIZE 
)

f1_history = []
f2_history = []
f3_history = []

def handle_exit(arg1, arg2):
    print("Exiting!")

    f1_stats = np.quantile(np.array(f1_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f2_stats = np.quantile(np.array(f2_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    # f3_stats = np.quantile(np.array(f3_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])

    print(f"{f1_stats}")    
    print(f"{f2_stats}")
    # print(f"{f3_stats}")

    visualizer.destroy()
    audio_stream.stop_stream()
    audio_stream.close()
    pyaud.terminate()

    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

audio_stream.start_stream()
sample = -1
alpha = 0.5




while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            handle_exit(1,2)

    #read in a window of 1024 samples at a time
    data = audio_stream.read(BUFFER_SIZE)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)

    #https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    
    #pre-emph
    # samples = np.append(samples[0], samples[1:] - alpha * samples[:-1])
    hamming = np.hamming(len(samples))
    samples = samples * hamming

    #one per 1000hz sampling rate
    samples_lpc = librosa.lpc(y=samples, order=44)

    # Find the roots of the LPC coefficients
    roots = np.roots(samples_lpc)
    roots = [r for r in roots if np.imag(r) >= 0]

    # Angular frequency
    ang_freq = np.angle(roots)


    formants = sorted(ang_freq * (SAMPLING_RATE / (2 * np.pi)))

    if formants[0] == 0:
        formants[0] = formants[1]
        formants[1] = formants[2]
        formants[2] = formants[3]

        if formants[0] == 0:
            formants[0] = formants[1]
            formants[1] = formants[2]

    f1, f2 = formants[0], formants[1]

    f1_smoother.append(f1)
    f2_smoother.append(f2)

    f1 = np.median(f1_smoother)
    f2 = np.median(f2_smoother)
    
    f1_history.append(f1)
    f2_history.append(f2)

    print(f"f1: {int(f1)} f2: {int(f2)}")


    visualizer.draw_formant(f2, f1)

    
