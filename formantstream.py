import numpy as np
import librosa

class FormantStream():
    def __init__(self, voicestream, f1_smoother, f2_smoother):
        self.voicestream = voicestream
        
        self.f1_smoother = f1_smoother
        self.f2_smoother = f2_smoother
    
    #https://stackoverflow.com/questions/61519826/how-to-decide-filter-order-in-linear-prediction-coefficients-lpc-while-calcu
    def get_formants(self, number_of_formants):
        
        data = self.voicestream.read_buffer()
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        order = self.voicestream.SAMPLING_RATE // 1000
        # order = 22
        alpha = 0.9
        
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

        # if formants[0] == 0:
        #     formants[0] = formants[1]

        # formants[0] = formants[1] if formants[0] == 0 else formants [0]

        # print("Foo")
        new_f1 = self.f1_smoother.insert(formants[0])
        new_f2 = self.f2_smoother.insert(formants[1])

        return [new_f1, new_f2]
        # return formants[:number_of_formants]
