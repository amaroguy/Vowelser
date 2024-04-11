import pyaudio
import wave

# Audio recording parameters
FORMAT = pyaudio.paInt16 # Data format
CHANNELS = 1 # Mono audio
RATE = 44100 # Sample rate
CHUNK = 1024 # Number of frames per buffer
RECORD_SECONDS = 5 # Length of recording
WAVE_OUTPUT_FILENAME = "output.wav" # Output filename

# Initialize pyaudio
audio = pyaudio.PyAudio()

# Open stream
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record for 5 seconds
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
audio.terminate()

# Save the recorded data as a WAV file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()