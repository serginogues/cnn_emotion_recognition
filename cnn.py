"""
CNN architecture with PyTorch
https://kaunild.github.io/experiments/experiments-1/
https://jovian.ai/himani007/logistic-regression-fer
"""
from preprocess import *

input_size = 48 * 48

def accuracy(output, labels):
    predictions, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        #ToDo: exepriment with this
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
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),  # 22x22x32 -> Out = 18x18x64 where 18=22-5+1 and 64=32*2
            nn.MaxPool2d(kernel_size=(2, 2)),  # 19x19x64 -> O = 9x9x64
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),  # 9x9x64 -> Out = 5x5x128 where 5=9-5+1 and 128=64*2
            nn.MaxPool2d(kernel_size=(2, 2)),  # 5x5x128 -> O = 2x2x128
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 2 * 128, 500),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, 7),  # 7 classes output
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # x = x.view(-1, 2 * 2 * 128)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        y = self.fc2(x)
        return y

    def train(self, epochs, optimizer, criterion, train_dataloader, test_dataloader):
        loss_history = []
        acc_history = []
        for epoch in range(epochs):
            print('Epoch ', epoch+1, 'of ', epochs)
            count = 0 
            for X, label in train_dataloader:
                pred = self.forward(X) 
                J = criterion(pred, label) 
                
                optimizer.zero_grad() 
                J.backward()
                optimizer.step() #backpropagate
                print(' Batch -> ',count)
                count += 1
            for X, label in test_dataloader:
                #calculate the accuracy on test set and print
                pred = self.forward(X)
                J = criterion(pred, label) 
                acc = accuracy(pred, label)
                print(' Loss: ', J)
                loss_history.append(J)
                print(' Accuracy: ', acc)
                acc_history.append(acc)

        return loss_history, acc_history

cnn = CNN()
cnn = cnn.to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer = optim.Adadelta(cnn.parameters(), lr=LR, rho=0.95, eps=1e-08)
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)

#ToDo: train function, test, evaluate
cnn.train(2, optimizer, criterion, train_loader, test_loader)