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






#PYAUDIO SETUP
SAMPLING_RATE = 44100
BUFFER_SIZE = 8192 ## of 2 byte samples

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

#handling button presses
def press_button(button):
    gamepad.press_button(button=button)
    print("oh lord im pressing")
    gamepad.update()
    time.sleep(0.5)
    print("releasing")
    gamepad.release_button(button=button)
    gamepad.update()

#LOAD MODEL
model = load("lr_dump.joblib")

#transform the incoming audio for the model
def build_features(samples):

    samples = samples * np.hanning(len(samples))
    mfcc = np.mean(librosa.feature.mfcc(y=samples, n_mfcc=40, sr=44100).T, axis=0)
    mel_spec = np.mean(librosa.feature.melspectrogram(y=samples, sr=44100).T, axis=0)

    return np.concatenate((mfcc, mel_spec), axis=0)



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
        #read in a window of 4096 samples at a time
        data = audio_stream.read(BUFFER_SIZE)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        feat = build_features(samples)
        feat =feat.reshape(1,-1)
        pred = model.predict(feat)[0]
        

        if pred != "silence":
            print(pred) 

        if last_button_pressed == pred:
            continue

        match pred:
            case "p":
                print("REE")
                press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            case "dj":
                press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
            case "k":
                press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            case _:
                pass


except Exception as e:
    print("Oh no", e)
    print("CTRL C")
finally:
    handle_exit(1,2)