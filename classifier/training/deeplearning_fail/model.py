from torch import nn
from torchinfo import summary

class PTK_CNNNetwork(nn.Module):

    #(1,24,9)
    def __init__(self):
        super().__init__()
        #3 conv blocks / flatten / linear / softmax

        #input -> (1, 513, 44) pad -> (1, 517, 48) conv -> (16, 515, 46) maxpool -> (16, 257, 23)
        #input -> (1, 65, 44) pad -> (1, 68, 48) conv -> (16, 66, 46) maxpool (16 , 33, 23)
        #input -> (1,24,17) pad -> (1, 28, 21) conv -> (16,26,19) maxpool -> (16,13,9)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #(input -> (16, 257, 23) pad -> (16, 261, 27) conv -> (32, 259, 25) pool -> (32, 129, 12)
        #input -> (16, 33, 23) pad -> (16, 37, 27) conv -> (64, 35, 25) -> (64, 17, 12)
        #input -> (16, 13, 9) pad -> (1, 17, 13) conv -> (32, 15, 11) maxpool -> (32, 7, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #input -> (32, 129, 12) padded -> (32, 133, 16) conv -> (64, 131, 14) pool -> (64, 65, 7)
        #input -> (32, 17, 12) padded -> (32, 21, 16) conv -> (64, 19, 14) pool -> (64, 9, 7) 
        #input -> (32,7,5): padded -> (32,11,9): conv -> (64,9,7): output -> (64, 4, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #input -> (64, 65, 7) padding -> (64, 69, 11) conv -> (128, 67, 9) pool -> (128, 33, 4)
        #input -> (64, 9, 7) padding -> (64, 13, 11) conv -> (128, 11, 9) pool -> (128, 5, 4)
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features= 128 * 33 * 4, out_features=4)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, input_data):
        # print(f"Current shape: {input_data.shape}")
        x = self.conv1(input_data)
        # print(f"Current shape after conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"Current shape after conv2: {x.shape}")
        x = self.conv3(x)
        # print(f"Current shape after conv3: {x.shape}")
        x = self.conv4(x)
        # print(f"Current shape after conv4 AAAAAAAAAAA: {x.shape}")
        x = self.flatten(x)
        # print(f"Current shape after flatten: {x.shape}")
        logits = self.linear(x)
        predictions = self.softmax(logits)

        return predictions
    
if __name__ == "__main__":
    cnn = PTK_CNNNetwork()
    summary(cnn, input_size=(1,513,44), batch_dim=1)
    