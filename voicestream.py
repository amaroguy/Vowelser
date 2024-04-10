import pyaudio as pyaud

class VoiceStream():
    
    def __init__(self, SAMPLING_RATE, BUFFER_SIZE):
        
        self.SAMPLING_RATE = SAMPLING_RATE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.pyaud = pyaud.PyAudio()
        self.audio_stream = self.pyaud.open(
                rate= SAMPLING_RATE, 
                channels=1,
                format=pyaud.get_format_from_width(2, True), #bit depth!
                input=True,
                output=False,
                input_device_index=0,
                frames_per_buffer=BUFFER_SIZE 
            )
        
        self.audio_stream.start_stream()
        
    def destroy(self):
        self.pyaud.terminate()
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        
    def read_buffer(self):
        return self.audio_stream.read(self.BUFFER_SIZE)
        
    
        