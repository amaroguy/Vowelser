import pyaudio
import wave
import numpy as np

# Configuration
SAMPLING_RATE = 44100
BUFFER_SIZE = 22050  # number of samples per buffer

# Open the wave file
wav_path = '.\\data\\k\\k_1.wav'  # Replace with your WAV file path
wav_file = wave.open(wav_path, 'rb')

# Setup PyAudio
pyaud = pyaudio.PyAudio()
audio_stream = pyaud.open(
    format=pyaud.get_format_from_width(wav_file.getsampwidth(), unsigned=False),
    channels=wav_file.getnchannels(),
    rate=wav_file.getframerate(),
    input=True,
    output=False,
    frames_per_buffer=BUFFER_SIZE
)

# Reading and processing the audio file
try:
    while True:
        data = wav_file.readframes(BUFFER_SIZE)
        if not data:
            break  # Stop if there are no more frames to read

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
        print("Buffered samples:", samples)

except Exception as e:
    print("An error occurred:", e)

finally:
    # Clean up and close everything properly
    audio_stream.stop_stream()
    audio_stream.close()
    pyaud.terminate()
    wav_file.close()

print("Finished reading the audio file.")