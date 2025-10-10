import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random


# load training and test data
def loadData():
    X_train = np.load('X_train.npy',allow_pickle=True)
    y_train = np.load('y_train.npy',allow_pickle=True)
    X_test = np.load('X_test.npy',allow_pickle=True)
    y_test = np.load('y_test.npy',allow_pickle=True)

    X_train = [torch.Tensor(x) for x in X_train]  # List of Tensors (SEQ_LEN[i],INPUT_DIM) i=0..NUM_SAMPLES-1
    X_test = [torch.Tensor(x) for x in X_test]  # List of Tensors (SEQ_LEN[i],INPUT_DIM)
    y_train = torch.Tensor(y_train) # (NUM_SAMPLES,1)
    y_test = torch.Tensor(y_test) # (NUM_SAMPLES,1)

    return X_train, X_test, y_train, y_test



# Define a Vanilla RNN layer by hand
class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.activation = torch.tanh

    def forward(self, x, hidden):
        hidden = self.activation(x @ self.W_xh + hidden @ self.W_hh)
        return hidden

# Define a sequence prediction model using the Vanilla RNN
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = RNNLayer(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        batch_size = len(input_seq)
        last_hidden = torch.zeros(batch_size, self.hidden_size, device=device)

        for b in range(batch_size):
            hidden = torch.zeros(self.hidden_size, device=device)

            seq_length =  seq_lengths[b]  

            for t in range(seq_length):
                hidden = self.rnn(input_seq[b][t], hidden)
            
            # Store the last hidden state in the output tensor
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output

# Define a sequence prediction model for fixed length sequences, BUT NO SHARED WEIGHTS
class SequenceModelFixedLen(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(SequenceModelFixedLen, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn_layers = nn.ModuleList([RNNLayer(input_size, hidden_size) for _ in range(seq_len)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths):
        batch_size = len(input_seq)
        last_hidden = torch.zeros(batch_size, self.hidden_size, device=device)

        for b in range(batch_size):
            hidden = torch.zeros(self.hidden_size, device=device).to(device)

            seq_length = min(self.seq_len, seq_lengths[b]) ######################################## I think???           
            for t in range(seq_length):
                hidden = self.rnn_layers[t](input_seq[b][t], hidden)
            
            # Store the last hidden state in the output tensor
            last_hidden[b] = hidden

        output = self.linear(last_hidden)
        return output



class PaddedModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len_max):
        super(PaddedModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len_max = seq_len_max
        self.rnn_layers = nn.ModuleList([RNNLayer(input_size, hidden_size) for _ in range(seq_len_max)])
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, padded_batch, lengths):
        B, T, _ = padded_batch.shape
        device = padded_batch.device

        hidden = [torch.zeros(self.hidden_size, device=device) for _ in range(B)]

        for t in range(T):
            for b in range(B):
                if t < lengths[b]:

                    hidden[b] = self.rnn_layers[t](padded_batch[b, t], hidden[b])

        last_hidden = torch.stack(hidden, dim=0)
        return self.linear(last_hidden)
            ####i think i have to basically do what was done above but again

# Define hyperparameters and other settings
input_size = 10  # Replace with the actual dimension of your input features
hidden_size = 64
output_size = 1
num_epochs = 10
learning_rate = 0.001
batch_size = 32


# load data
X_train, X_test, y_train, y_test = loadData()
device = y_train.device

# Create the model using min length input
seq_lengths = [seq.shape[0] for seq in X_train]



# Training loop
def train(model, num_epochs, lr, batch_size, X_train, y_train, seq_lengths):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("training!")
    for epoch in range(num_epochs):

        print("epoch ", epoch)

        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]
            lengths = seq_lengths[i:i+batch_size]

            #GPU related stuff to ensure it picks the right device
            inputs  = [x.to(device) for x in inputs]
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(loss)
    return model

def train_padded(model, num_epochs, lr, batch_size, X_train, y_train):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("training padded!")

    for epoch in range(num_epochs):
        print("epoch ",epoch)
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size].to(device)

            lengths = [len(s) for s in batch]
            padded = pad_sequence(batch, batch_first=True).to(device)
            optimizer.zero_grad()
            outputs = model(padded, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(loss.item())   



# initialize and train Vanilla RNN
if __name__ == "__main__":
    ################################################################################################needs revision
    X_train, X_test, y_train, y_test = loadData()

    if torch.cuda.is_available():
        device = torch.device("cuda") # pick my gpu
        print("cuda selected!")
    else:
        device = torch.device("cpu")
        print("cpu selected. no visible gpu")

    print("Vanilla RNN . . . . .")
    vanilla = SequenceModel(input_size, hidden_size, output_size).to(device)
    train_vanilla_RNN =train(vanilla, num_epochs, learning_rate, batch_size, X_train, y_train, seq_lengths)
   

    print ("fixed length truncated model....")
    

    Lmin = min(seq_lengths)
    X_train_trunc = []

    for x in X_train:
         truncated_seq = x[:Lmin]
         X_train_trunc.append(truncated_seq) 

    seq_lengths_trunc = [Lmin] * len(X_train_trunc)


    trunc = SequenceModelFixedLen(input_size, hidden_size, output_size, seq_len=Lmin).to(device)
    Train_trunc = train(trunc, num_epochs, learning_rate, batch_size, X_train_trunc, y_train, seq_lengths_trunc)



    print("padded model ....")
    Lmax = max(seq_lengths)
    padded_model = PaddedModel(input_size, hidden_size, output_size, seq_len_max=Lmax).to(device)
    train_padded(padded_model, num_epochs, learning_rate, batch_size, X_train, y_train)











###################################################################################################################################

# initialize and train Sequential NN fixing #timesteps to the minimum sequence length

# initialize and train Sequential NN fixing #timesteps to the maximum sequence length
# NOTE: it is OK to use torch.nn.utils.rnn.pad_sequence; make sure to set parameter batch_first correctly
