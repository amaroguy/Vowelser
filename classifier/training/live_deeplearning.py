import signal
import sys
import pyaudio
import numpy as np
import librosa
from model import PTK_CNNNetwork
import torchaudio
import torch

#pyaudio setup
#We'll be taking in about .1 seconds of data every time we do LPC w/2048 samples
#because of the 2048 / sampling rate
SAMPLING_RATE = 44100
BUFFER_SIZE = 22050 ## of 2 byte samples

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

    audio_stream.stop_stream()
    audio_stream.close()
    pyaud.terminate()

    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)

audio_stream.start_stream()
sample = -1
alpha = 0.5

mel_spectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLING_RATE,
    n_fft=1024, #frame size
    hop_length=512, #how much does the frame slide over
    n_mels=64
)

class_mapping = [
    "p",
    "t",
    "k",
    "vowel"
]


def predict(model, input, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
    return predicted

model = PTK_CNNNetwork()
state_dict = torch.load("ptk_netreedo.pth")
model.load_state_dict(state_dict)

try:
    while True:
        #read in a window of 4096 samples at a time
        data = audio_stream.read(BUFFER_SIZE)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        # print(samples.shape)
        audio_signal_tensor = torch.Tensor(samples, device="cpu")
        mel = mel_spectogram(audio_signal_tensor)
        mel = torch.unsqueeze(mel, 0)
        mel = torch.unsqueeze(mel, 0)
        # print(mel.shape)

        prediction = predict(model, mel, class_mapping)

        if prediction != "vowel":
            print(prediction)

        # print(samples[:10])

except Exception as e:
    print("CTRL C")
finally:
    handle_exit(1,2)
        