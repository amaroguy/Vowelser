import torch
import torchaudio

from model import PTK_CNNNetwork
from PTKDataset import PTKDataset
from training import AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "p",
    "t",
    "k",
    "vowel"
]
#foo

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # load back the model
    cnn = PTK_CNNNetwork()
    state_dict = torch.load("ptk_netreedo.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    spectogram = torchaudio.transforms.Spectrogram(n_fft=1024,
                                                   hop_length=512)

    ptkd = PTKDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            NUM_SAMPLES,
                            spectogram)


    # get a sample from the urban sound dataset for inference

    correct = 0
    incorrect = 0

    for input, target in ptkd:
        input.unsqueeze_(0)
        print(input.shape)
        predicted, expected = predict(cnn, input, target,
                                    class_mapping)
        
        if predicted == expected:
            correct += 1
        else:
            print(f"Predicted: '{predicted}', expected: '{expected}'")
            incorrect += 1

    
    print((correct / len(ptkd)) * 100, "acc")

    # make an inference
    #/l/ 