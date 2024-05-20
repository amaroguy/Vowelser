import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader

from model import PTK_CNNNetwork
from PTKDataset import PTKDataset

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "C:\\Users\\Erick\\Desktop\\LIGN 168\\classifier\\training\\ptk.csv"
AUDIO_DIR = "C:\\Users\\Erick\\Desktop\\LIGN 168\\classifier\\training"
SAMPLE_RATE = 44100
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimiser, device=None):
    for input, target in data_loader:

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    spectogram = torchaudio.transforms.Spectrogram(n_fft=1024,
                                            hop_length=512)

    usd = PTKDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            NUM_SAMPLES,
                            spectogram)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    cnn = PTK_CNNNetwork()
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "ptk_netreedo.pth")
    print("Done!")