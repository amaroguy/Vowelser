import signal
import sys
import pyaudio
import numpy as np
import librosa
import vgamepad as vg
import time
from joblib import load
from sklearn.linear_model import LogisticRegression
from enum import Enum
from collections import deque

#PYAUDIO SETUP
SAMPLING_RATE = 44100
BUFFER_SIZE = 4096 ## of 2 byte samples

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

#CONTROLLER SETUP
gamepad = vg.VX360Gamepad()
BUTTON_DELAY = 0.5
BUTTONS_BUFFER = deque(maxlen=8192)

#LOAD MODEL
model = load("lr_dump.joblib")

#transform the incoming audio for the model
def build_features(samples):

    samples = samples * np.hanning(len(samples))
    mfcc = np.mean(librosa.feature.mfcc(y=samples, n_mfcc=40, sr=44100).T, axis=0)
    mel_spec = np.mean(librosa.feature.melspectrogram(y=samples, sr=44100).T, axis=0)

    return np.concatenate((mfcc, mel_spec), axis=0)

#CONTROL STICK SMOOTHING
SMOOTHTIME = 10
f1_smoother = deque(maxlen=SMOOTHTIME)
f2_smoother = deque(maxlen=SMOOTHTIME)

#CONTROL STICK LIMS
F2_RANGE = (500, 2800)
F1_RANGE = (100, 800)


#HANDLE EXIT
def handle_exit(arg1, arg2):
    print("Exiting!")

    audio_stream.stop_stream()
    audio_stream.close()
    pyaud.terminate()

    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

audio_stream.start_stream()
sample = -1
last_button_pressed = ""

try:
    while True:

        #READING DATA
        data = audio_stream.read(BUFFER_SIZE)
        samples_sticks = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        samples_buttons = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        BUTTONS_BUFFER += samples_buttons.tolist()
        # print(len(BUTTONS_BUFFER))
        #FORMANT CALCULATION
        #https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
        #pre-emph
        hamming = np.hamming(len(samples_sticks))
        samples_sticks = samples_sticks * hamming

        #one per 1000hz sampling rate
        samples_lpc = librosa.lpc(y=samples_sticks, order=44)

        # Find the roots of the LPC coefficients
        roots = np.roots(samples_lpc)
        roots = [r for r in roots if np.imag(r) >= 0]

        # Angular frequency
        ang_freq = np.angle(roots)

        ##something fishy going on here
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

        # print(f1)
        # print(f2)

        #BUTTON PREDICTION - only updates every 0.1 seconds
        if len(BUTTONS_BUFFER) == 8192:
            feat = build_features(BUTTONS_BUFFER)
            BUTTONS_BUFFER.clear()
            feat =feat.reshape(1,-1)
            pred = model.predict(feat)[0]
            
            # print(pred) 

            if last_button_pressed == pred:
                print("continuing!")
                continue
            elif last_button_pressed != "":
                gamepad.release_button(last_button_pressed)
                gamepad.update()

            match pred:
                case "p":
                    gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                    last_button_pressed = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
                case "dj":
                    gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                    last_button_pressed = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
                case "k":
                    gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
                    last_button_pressed = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
                case _:
                    pass
                
            gamepad.update()

        #JOYSTICK INPUT
        if f1 < F1_RANGE[0] or f1 > F1_RANGE[1]:
            print(f"f1: {f1} not in range ${F1_RANGE}")
            continue
        if f2 < F2_RANGE[0] or f2 > F2_RANGE[1]:
            print(f"f2: {f1} not in range ${F2_RANGE}")
            continue

        #TODO YOU SWAPPED THESE
        y = np.interp(f1, [F1_RANGE[0], F1_RANGE[1]], [1.0, -1.0])  
        x = np.interp(f2, [F2_RANGE[0], F2_RANGE[1]], [1.0, -1.0])
        
        gamepad.left_joystick_float(x_value_float=x, y_value_float=y)
        gamepad.update()
                    


except Exception as e:
    print("Oh no", e)
finally:
    handle_exit(1,2)