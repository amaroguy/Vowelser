import numpy as np
import librosa

class FormantStream():
    def __init__(self, voicestream, smoother):
        self.voicestream = voicestream
        
        #TODO
        self.smoother = smoother
    
    #https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    def get_formants(self, number_of_formants):
        
        data = self.voicestream.read_buffer()
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        order = self.voicestream.SAMPLING_RATE // 1000
        alpha = 0.7
        
        #pre-emph
        samples = np.append(samples[0], samples[1:] - alpha * samples[:-1])
        
        #one per 1000hz sampling rate
        samples_lpc = librosa.lpc(y=samples, order=order)

        # Find the roots of the LPC coefficients
        roots = np.roots(samples_lpc)
        roots = [r for r in roots if np.imag(r) >= 0]

        # Angular frequency
        ang_freq = np.angle(roots)
        
        formants = sorted(ang_freq * (self.voicestream.SAMPLING_RATE / (2 * np.pi)))
        
        # print("Foo")
        return formants[:number_of_formants]
