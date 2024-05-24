from torch.utils.data import Dataset
import torchaudio
import pandas
import os
import torch

torch.set_printoptions(edgeitems=50, linewidth=200, precision=10, threshold=5000)


#lots of this is adapted w;/ some tweaks from 
#https://github.com/musikalkemist/pytorchforaudio/blob/main/04%20Creating%20a%20custom%20dataset/urbansounddataset.py
class PTKDataset(Dataset):

    #Assumed to have 44100Hz sample rate
    def __init__(self, annotations_file, audio_dir, num_samples, transformation):
        self.annotations = pandas.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.transformation = transformation

    def __len__(self):
        return len(self.annotations)

    #dataset[x] -> get the data and label for this index
    def __getitem__(self, index):
        audio_sample_filepath = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_filepath, normalize=True)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, label

    def _get_audio_sample_path(self, index):
        path = self.annotations.iloc[index, 0]

        return os.path.join(self.audio_dir, path)

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 2]
        return label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples) #0 in the y dir, and the rest in the x dir
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

if __name__ == "__main__":
    AUDIO_PATH = "C:\\Users\\Erick\\Desktop\\LIGN 168\\classifier\\training"
    SAMPLE_RATE = 44100
    NUM_SAMPLES = 22050
    
    #PYTORCH IS DIVIDING BY 2**32
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024, #frame size
        hop_length=512, #how much does the frame slide over
        n_mels=64
    )

    spectogram = torchaudio.transforms.Spectrogram(n_fft=1024,
                                                   hop_length=512)

    mydataset = PTKDataset("ptk.csv", AUDIO_PATH, NUM_SAMPLES, spectogram)
    foo = mydataset._get_audio_sample_label(1)
    bar = mydataset._get_audio_sample_path(1)
    res = mydataset[1]

    print(foo)
    print(bar)
    print(res[0].shape)
