"""
https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d
"""

from config import *


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),  # Out = 48x48x64 48=48-5+2*2+1 where K=64
            nn.ELU(),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # Out = 48x48x64 and K=32
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Out = 48x48x128 and K=32
            nn.ELU(),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Out = 48x48x128 and K=32
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Out = 48x48x256 and K=32
            nn.ELU(),
            nn.BatchNorm2d(256)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Out = 48x48x256 and K=32
            nn.ELU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.4)
        )

        self.flat = nn.Sequential(
            nn.Flatten()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(6 * 6 * 256, 128),
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.6)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, NUM_LABELS),  # 7 classes output
            # nn.LogSoftmax(dim=1) # IMPORTANT: No Softmax must be applied in the last layer if we use the Cross-Entropy Loss
        )

    def forward(self, x):
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)
        x = self.conv6(x)
        #print(x.shape)
        x = self.flat(x)
        #print(x.shape)
        # x = x.view(-1, 2 * 2 * 128)
        #x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #print(x.shape)
        y = self.fc2(x)
        #print(y.shape)
        return y


class simple_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        #TODO: exepriment with this
        # we can try using
        # nn.PReLU(): https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
        # nn.ZeroPad2d(output_dim): https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html
        # nn.BatchNorm2d(output_dim)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),  # Out = 44x44x32 where 44=48-5+1 and K=32
            nn.MaxPool2d(kernel_size=(2, 2)),  # Out = 23x23x32
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            # 22x22x32 -> Out = 18x18x64 where 18=22-5+1 and 64=32*2
            nn.MaxPool2d(kernel_size=(2, 2)),  # 19x19x64 -> O = 9x9x64
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            # 9x9x64 -> Out = 5x5x128 where 5=9-5+1 and 128=64*2
            nn.MaxPool2d(kernel_size=(2, 2)),  # 5x5x128 -> O = 2x2x128
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 2 * 128, 500),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, NUM_LABELS),  # 7 classes output
            # nn.LogSoftmax(dim=1) # IMPORTANT: No Softmax must be applied in the last layer if we use the Cross-Entropy Loss
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # x = x.view(-1, 2 * 2 * 128)
        x = x.view(x.size(0), -1)  # x.size() = torch.Size([15, 512])
        x = self.fc1(x)
        y = self.fc2(x)
        return y


class LeNet5(nn.Module):
    """
    https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
    Layer 1 (C1): The first convolutional layer with 6 kernels of size 5×5 and the stride of 1. Given the input size (32×32×1), the output of this layer is of size 28×28×6.
    Layer 2 (S2): A subsampling/pooling layer with 6 kernels of size 2×2 and the stride of 2. The subsampling layer in the original architecture was a bit more complex than the traditionally used max/average pooling layers. I will quote [1]: “ The four inputs to a unit in S2 are added, then multiplied by a trainable coefficient, and added to a trainable bias. The result is passed through a sigmoidal function.”. As a result of non-overlapping receptive fields, the input to this layer is halved in size (14×14×6).
    Layer 3 (C3): The second convolutional layer with the same configuration as the first one, however, this time with 16 filters. The output of this layer is 10×10×16.
    Layer 4 (S4): The second pooling layer. The logic is identical to the previous one, but this time the layer has 16 filters. The output of this layer is of size 5×5×16.
    Layer 5 (C5): The last convolutional layer with 120 5×5 kernels. Given that the input to this layer is of size 5×5×16 and the kernels are of size 5×5, the output is 1×1×120. As a result, layers S4 and C5 are fully-connected. That is also why in some implementations of LeNet-5 actually use a fully-connected layer instead of the convolutional one as the 5th layer. The reason for keeping this layer as a convolutional one is the fact that if the input to the network is larger than the one used in [1] (the initial input, so 32×32 in this case), this layer will not be a fully-connected one, as the output of each kernel will not be 1×1.
    Layer 6 (F6): The first fully-connected layer, which takes the input of 120 units and returns 84 units. In the original paper, the authors used a custom activation function — a variant of the tanh activation function. For a thorough explanation, please refer to Appendix A in [1].
    Layer 7 (F7): The last dense layer, which outputs 10 units. In [1], the authors used Euclidean Radial Basis Function neurons as activation functions for this layer.
    """

    def __init__(self, n_classes=7):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1), # Out = 44x44x32 where 44=48-5+1 and K=32
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=3000, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)  # x.size() = torch.Size([15, 3000])
        logits = self.classifier(x)
        # probs = F.softmax(logits, dim=1)
        return logits




