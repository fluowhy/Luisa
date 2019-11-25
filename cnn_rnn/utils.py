import torch
from torch.utils.data import Dataset, DataLoader
import random
import os
import numpy as np
import pdb
import pandas as pd


def read_hyperparameters(savename):
    params_df = pd.read_csv(savename)
    names = params_df.columns
    params = {}
    for name in names:
        params[name] = float(params_df[name].values[0])
    return params


def save_hyperparamters(names, values, savename):
    df = {}
    for i, name in enumerate(names):
        df[name] = [values[i]]
    df = pd.DataFrame(data=df)
    df.to_csv(savename, index=False)
    return


def seed_everything(seed=1234):
    """
    Author: Benjamin Minixhofer
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.n, sl, _ = x.shape  # rnn
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)
        self.z = torch.tensor(z, dtype=torch.long)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]

    def __len__(self):
        return self.n


class LuisaDataset(object):
    def __init__(self, bs):
        # shape nsamples, time_steps, latent_dim
        self.x_train = np.load("../data/processed/x_train.npy")
        self.x_val = np.load("../data/processed/x_val.npy")

        self.y_train = np.load("../data/processed/y_train.npy")
        self.y_val = np.load("../data/processed/y_val.npy")

        self.seq_len_train = np.load("../data/processed/seq_len_train.npy")
        self.seq_len_val = np.load("../data/processed/seq_len_val.npy")

        self.bs = bs
        self.process_all()    

    def define_datasets(self):
        self.train_dataset = MyDataset(self.x_train, self.y_train, self.seq_len_train)
        self.val_dataset = MyDataset(self.x_val, self.y_val, self.seq_len_val)
        return

    def define_dataloaders(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.bs, shuffle=True)
        return

    def process_all(self):
        self.define_datasets()
        self.define_dataloaders()
        return


def count_parameters(model):
    # TODO: add docstring
    """
    Parameters
    ----------
    model
    Returns
    -------
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Label2OneHot(object):
    def __init__(self, nlabels):
        self.nlabels = nlabels

    def __call__(self, x):
        return torch.nn.functional.one_hot(x.long(), num_classes=self.nlabels).float()
        