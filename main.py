import signal
import sys
import pyaudio
import numpy as np
import librosa
import pygame
from visualize import FormantTracker
import statistics

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

#handle ctrl c

f1_history = []
f2_history = []
f3_history = []

def handle_exit(arg1, arg2):
    print("Exiting!")

    f1_stats = np.quantile(np.array(f1_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f2_stats = np.quantile(np.array(f2_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])
    f3_stats = np.quantile(np.array(f3_history).astype(np.int32) , [0,0.25, 0.5, 0.75,1])

    print(f"{f1_stats}")    
    print(f"{f2_stats}")
    print(f"{f3_stats}")

    visualizer.destroy()
    audio_stream.stop_stream()
    audio_stream.close()
    pyaud.terminate()

    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

audio_stream.start_stream()
sample = -1
alpha = 0.9




while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            handle_exit(1,2)

    #read in a window of 1024 samples at a time
    data = audio_stream.read(BUFFER_SIZE)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)

    #https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    
    #pre-emph
    samples = np.append(samples[0], samples[1:] - alpha * samples[:-1])

    #one per 1000hz sampling rate
    samples_lpc = librosa.lpc(y=samples, order=44)

    # Find the roots of the LPC coefficients
    roots = np.roots(samples_lpc)
    roots = [r for r in roots if np.imag(r) >= 0]

    # Angular frequency
    ang_freq = np.angle(roots)

    # Convert from rad/sample to Hz
    #idea: if f1 > 0 and f2 > 0?

    ##something fishy going on here
    formants = sorted(ang_freq * (SAMPLING_RATE / (2 * np.pi)))
    f1, f2, f3, f4 = formants[:4]
    f1_history.append(f1) 
    f2_history.append(f2)
    f3_history.append(f3)
    print(f"f1: {int(f1)} f2: {int(f2)}, f3: {int(f3)} f4: {f4}")


    visualizer.draw_formant(f3, f2)

    