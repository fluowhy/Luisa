import torch
import pdb


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class IMG2SEQ(torch.nn.Module):
    def __init__(self, nin, nh, nlayers=1, ks=3, nf=2, do=0.5):
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
        self.conv2 = torch.nn.Conv2d(self.nf, 2 * self.nf, kernel_size=3, padding=int(ks / 2))
        self.conv3 = torch.nn.Conv2d(2 * self.nf, 4 * self.nf, kernel_size=3, padding=int(ks / 2))
        self.conv4 = torch.nn.Conv2d(4 * self.nf, 8 * self.nf, kernel_size=3, padding=int(ks / 2))
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=2)
        self.bn1 = torch.nn.BatchNorm2d(self.nf)       
        self.bn2 = torch.nn.BatchNorm2d(2 * self.nf)       
        self.bn3 = torch.nn.BatchNorm2d(4 * self.nf)
        self.bn4 = torch.nn.BatchNorm2d(8 * self.nf)
        self.flatten = Flatten()

        self.fc = torch.nn.Linear(nh, int((nin + nh) * 0.5))
        self.out = torch.nn.Linear(int((nin + nh) * 0.5), nin)
        self.bn = torch.nn.BatchNorm1d(int((nin + nh) * 0.5))

    def convolutions(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpool3(self.relu(self.bn3(self.conv3(x))))
        x = self.maxpool4(self.relu(self.bn4(self.conv4(x))))
        return self.flatten(x)

    def forward(self, img, text):
        f = self.convolutions(img)
        y_pred = self.recurrence(text, f)
        return y_pred

    def recurrence(self, x, hidden):
        x, _ = self.rnn(x, hidden.unsqueeze(0))
        x = self.fc(x)
        x = self.bn(x.transpose(1, 2))
        x = self.relu(x.transpose(1, 2))
        x = self.out(x)
        return x

    def predict(self, img, start_char, end_char, lim=1):        
        hidden = self.convolutions(img.unsqueeze(1)).unsqueeze(0)
        xi = start_char
        encoded_word = [start_char]
        x = torch.zeros(self.nin)
        x[start_char] = 1
        x = x.unsqueeze(0).unsqueeze(1)
        while xi != end_char and len(encoded_word) < lim:
            x, hidden = self.rnn(x, hidden)
            x = self.out(self.relu(self.bn(self.fc(x).transpose(1, 2)).transpose(1, 2)))
            xi = torch.argmax(x.squeeze())
            encoded_word.append(xi.cpu().numpy())
        return encoded_word


if __name__ == "__main__":
    from utils import *
    import numpy as np
    x = np.load("../data/processed/x.npy")
    y = np.load("../data/processed/y.npy")
    idx2char = np.load("../data/processed/idx2char.npy", allow_pickle=True)

    x = torch.tensor(x, dtype=torch.float, device="cpu").unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long, device="cpu")

    label2onehot = Label2OneHot(len(idx2char))
    y_oh = label2onehot(y)
    model = IMG2SEQ(nin=y_oh.shape[-1], nh=128, nlayers=1, ks=3, nf=2, do=0.5)

    model.eval()
    with torch.no_grad():
        y_pred = model(x, y_oh)
