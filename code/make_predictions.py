import torch
import numpy as np
import matplotlib.pyplot as plt

from models import *


def decode_word(encoded_word):
    decoded_word = []
    for let in encoded_word:
        dw = idx2char[let]
        if dw != "<eos>" and dw != "<pad>":
            decoded_word.append(idx2char[let])
    return "".join(decoded_word[1:])


def predit_and_plot(idx, sp):
    model.eval()
    with torch.no_grad():
        seq = model.predict(x_val[idx], char2idx["<sos>"], char2idx["<eos>"], lim=10)
    word = decode_word(seq)
    gt = decode_word(y_val[idx])
    plt.subplot(sp)
    plt.imshow(x_val[idx].squeeze())
    plt.title("gt: {} \n pred: {}".format(gt, word))
    return


device = "cpu"

x_val = np.load("../data/processed/x_val.npy")
y_val = np.load("../data/processed/y_val.npy")

idx2char = np.load("../data/processed/idx2char.npy", allow_pickle=True)
char2idx = np.load("../data/processed/char2idx.npy", allow_pickle=True).item()

model_name = "rnn_nin_110_nh_512_nlayers_1_ks_3_nf_8_do_0.5"
model = IMG2SEQ(nin=110, nh=512, nlayers=1, ks=3, nf=8, do=0.5)

model.load_state_dict(torch.load("../models/{}.pth".format(model_name), map_location=device))

x_val = torch.tensor(x_val, dtype=torch.float, device="cpu").unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.long, device="cpu")

l = 4
plt.clf()
plt.figure(figsize=(3 * l, 3 * l))
sp = 331
for i in range(9):
    predit_and_plot(i, sp)
    sp += 1
plt.show()