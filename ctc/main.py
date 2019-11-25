import argparse
import numpy as np
from tqdm import tqdm

from models import *
from utils import *


class Classifier(object):
    def __init__(self, model, nin, nout, nh, nlayers=1, ks=3, nf=2, do=0.5, device="cpu", pre=False):
        self.model_name = "{}_nin_{}_nout_{}_nh_{}_nlayers_{}_ks_{}_nf_{}_do_{}".format(model, nin, nout, nh, nlayers, ks, nf, do)
        self.device = device
        self.model = IMG2SEQ(nin, nout, nh, nlayers, ks, nf, do)
        self.model.to(device)
        self.best_loss = np.inf
        # save hyperparameters
        names = ["nin", "nout", "nh", "nlayers", "ks", "nf", "do"]
        values = [nin, nout, nh, nlayers, ks, nf, do]
        save_hyperparamters(names, values, "../files/params_{}.csv".format(self.model_name))
        self.load_model() if pre else 0
        print("model params {}".format(count_parameters(self.model)))
        idx2char = np.load("../data/processed/idx2char_ctc.npy", allow_pickle=True)
        char2idx = np.load("../data/processed/char2idx_ctc.npy", allow_pickle=True).item()  
        self.ctc = torch.nn.CTCLoss(blank=len(idx2char), reduction="mean", zero_infinity=False)

    def train_model(self, data_loader, clip_value=1.):
        self.model.train()
        train_loss = 0
        for idx, batch in tqdm(enumerate(data_loader)):
            self.optimizer.zero_grad()
            x, y, seq_len = batch
            x = x.to(self.device).unsqueeze(1)
            y = y.to(self.device)
            seq_len = seq_len.to(self.device)
            y_pred = self.model(x)
            input_lengths = torch.full(size=(x.size(0),), fill_value=y_pred.size(1), dtype=torch.long)
            loss = self.ctc(y_pred.transpose(0, 1), y, input_lengths, seq_len)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= (idx + 1)
        return train_loss

    def eval_model(self, data_loader):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(data_loader)):
                x, y, seq_len = batch
                x = x.to(self.device).unsqueeze(1)
                y = y.to(self.device)
                seq_len = seq_len.to(self.device)
                y_pred = self.model(x)
                input_lengths = torch.full(size=(x.size(0),), fill_value=y_pred.size(1), dtype=torch.long)
                loss = self.ctc(y_pred.transpose(0, 1), y, input_lengths, seq_len)
                eval_loss += loss.item()
        eval_loss /= (idx + 1)
        return eval_loss

    def fit(self, train_loader, val_loader, epochs, lr, wd):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
        template = "Epoch {} \n Train loss: {:.3f} Val loss l2: {:.3f}"
        # self.optimizer = RAdam(self.model.parameters(), lr=lr, weight_decay=wd)
        loss = []
        for epoch in range(epochs):
            train_loss = self.train_model(train_loader)
            val_loss = self.eval_model(val_loader)
            loss.append((train_loss, val_loss))
            print(template.format(epoch, train_loss, val_loss))
            if val_loss < self.best_loss:
                print("Saving")
                self.best_loss = val_loss
                self.save_model()
            np.save("../files/{}".format("loss"), np.array(loss))
        return

    def save_model(self):
        torch.save(self.model.state_dict(), "../models/{}.pth".format(self.model_name))
        return

    def load_model(self):
        self.model.load_state_dict(torch.load("../models/{}.pth".format(self.model_name), map_location=self.device))
        return


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser(description="auto encoder")
    parser.add_argument('--bs', type=int, default=100, help="batch size (default 100")
    parser.add_argument('--e', type=int, default=2, help="epochs (default 2)")
    parser.add_argument("--d", type=str, default="cpu", help="select device (default cpu)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default 2e-4)")
    parser.add_argument("--pre", action="store_true", help="train pre trained model (default False)")
    parser.add_argument('--nh', type=int, default=2, help="number of hidden units (default 2)")
    parser.add_argument('--nlayers', type=int, default=1, help="number of hidden layers (default 1)")
    parser.add_argument("--do", type=float, default=0., help="dropout value (default 0)")
    parser.add_argument("--wd", type=float, default=0., help="L2 reg value (default 0.)")
    parser.add_argument("--model", type=str, default="rnn", help="model type, options: rnn, gru, lstm (default rnn)")
    parser.add_argument('--ks', type=int, default=3, help="kernel size (default 3)")
    parser.add_argument('--nf', type=int, default=2, help="number of starting filters (default 2)")
    args = parser.parse_args()
    print(args)
    device = args.d

    luisa = LuisaDataset(args.bs)
    idx2char = np.load("../data/processed/idx2char_ctc.npy", allow_pickle=True)
    nin = 288
    nout = len(idx2char)

    rnn_model = Classifier(args.model, nin, nout, args.nh, args.nlayers, args.ks, args.nf, args.do, args.d, args.pre)
    rnn_model.fit(luisa.train_dataloader, luisa.val_dataloader, args.e, args.lr, args.wd)


