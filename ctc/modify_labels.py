import numpy as np
from pandas import read_csv


# Load transformations from label num to label char and viceversa
idx2char = np.load("../data/processed/idx2char.npy", allow_pickle=True)
char2idx = np.load("../data/processed/char2idx.npy", allow_pickle=True).item()

# Load labels
x = np.load("../data/processed/x.npy")
y = np.load("../data/processed/y.npy")
n, sl = y.shape

# Remove <sos> and <eos>
y = y[:, 1:]
y = y.reshape((-1, 1))
y = y[y != char2idx["<eos>"]].reshape((n, -1))

# Transform index to char
y_char = []
for i in range(y.shape[0]):
    converted_string = [idx2char[j] for j in y[i]]
    y_char.append(converted_string)

idx2char = np.unique(y_char)
char2idx = {u:i for i, u in enumerate(idx2char)}

# New char to index
y = []
for i in range(len(y_char)):
    converted_index = [char2idx[j] for j in y_char[i]]
    y.append(converted_index)
y = np.array(y)

# Calculate sequence length
seq_len = (y != char2idx["<pad>"]).sum(1)

# Save new transformations
np.save("../data/processed/idx2char_ctc.npy", idx2char)
np.save("../data/processed/char2idx_ctc.npy", char2idx)
np.save("../data/processed/y_ctc.npy", y)
np.save("../data/processed/seq_len_ctc.npy", seq_len)

# validation set
n = int(len(x) * 0.9)

np.save("../data/processed/x_train_ctc.npy", x[:n])
np.save("../data/processed/y_train_ctc.npy", y[:n])
np.save("../data/processed/seq_len_train_ctc.npy", seq_len[:n])
np.save("../data/processed/x_val_ctc.npy", x[n:])
np.save("../data/processed/y_val_ctc.npy", y[n:])
np.save("../data/processed/seq_len_val_ctc.npy", seq_len[n:])
