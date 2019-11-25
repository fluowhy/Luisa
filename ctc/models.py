import torch
import pdb


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class MixDIM(torch.nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), input.size(1) * input.size(2), -1)


class IMG2SEQ(torch.nn.Module):
    def __init__(self, nin, nout, nh, nlayers=1, ks=3, nf=2, do=0.5):
        super(IMG2SEQ, self).__init__()
        self.nin = nin
        self.nh = nh
        self.ks = ks
        self.relu = torch.nn.ReLU()
        self.nf = nf
        if nlayers >= 2:
            self.rnn = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.rnn = torch.nn.GRU(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(1, self.nf, kernel_size=3, padding=int(ks / 2))
        self.conv2 = torch.nn.Conv2d(self.nf, self.nf, kernel_size=3, padding=int(ks / 2))
        self.conv3 = torch.nn.Conv2d(self.nf, 2 * self.nf, kernel_size=3, padding=int(ks / 2))
        self.conv4 = torch.nn.Conv2d(2 * self.nf, 2 * self.nf, kernel_size=3, padding=int(ks / 2))
        # self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
        self.bn1 = torch.nn.BatchNorm2d(self.nf)       
        self.bn2 = torch.nn.BatchNorm2d(self.nf)       
        self.bn3 = torch.nn.BatchNorm2d(2 * self.nf)
        self.bn4 = torch.nn.BatchNorm2d(2 * self.nf)
        self.flatten = MixDIM()

        self.out = torch.nn.Linear(nh, nout + 1)

    def convolutions(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool4(self.relu(self.bn4(self.conv4(x))))
        x = self.flatten(x).transpose(1, 2)
        return x

    def forward(self, x):
        x = self.convolutions(x)
        x = self.recurrence(x)
        x = self.out(x)
        return x

    def recurrence(self, x):
        x, _ = self.rnn(x)
        return x

    def predict(self, x):        
        x = self.forward(x)
        return x


if __name__ == "__main__":
    from utils import *
    import numpy as np
    n = 100
    x = np.load("../data/processed/x.npy")[:n]
    y = np.load("../data/processed/y_ctc.npy")[:n]

    idx2char = np.load("../data/processed/idx2char_ctc.npy", allow_pickle=True)

    nin = 36
    nout = len(idx2char)    

    x = torch.tensor(x, dtype=torch.float, device="cpu").unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long, device="cpu")

    model = IMG2SEQ(nin=nin, nout=nout, nh=128, nlayers=1, ks=3, nf=2, do=0.5)

    model.eval()
    with torch.no_grad():
        y_pred = model(x)
