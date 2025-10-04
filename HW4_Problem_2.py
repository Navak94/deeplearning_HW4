import torch
import torch.nn as nn
import torch.optim as optim
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
        self.W_xh = 
        self.W_hh = 
        self.activation =

    def forward(self, x, hidden):
        hidden = 
        return hidden

# Define a sequence prediction model using the Vanilla RNN
class SequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = RNNLayer( , )
        self.linear = nn.Linear( , )

    def forward(self, input_seq, seq_lengths):
        batch_size = 
        last_hidden = torch.zeros( , ).to(device)

        for b in range(batch_size):
            hidden = torch.zeros( ).to(device)

            seq_length =             
            for t in range(seq_length):
                hidden = self.rnn(input_seq[ ][ ], hidden)
            
            # Store the last hidden state in the output tensor
            last_hidden[ ] = 

        output = self.linear( )
        return output

# Define a sequence prediction model for fixed length sequences, BUT NO SHARED WEIGHTS
class SequenceModelFixedLen(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(SequenceModelFixedLen, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.rnn_layers = [RNNLayer( , ) for _ in range( )]
        self.linear = nn.Linear( , )

    def forward(self, input_seq, seq_lengths):
        batch_size = 
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(device)

        for b in range(batch_size):
            hidden = torch.zeros( ).to(device)

            seq_length = min( , )            
            for t in range(seq_length):
                hidden = 
            
            # Store the last hidden state in the output tensor
            last_hidden[ ] = 

        output = self.linear()
        return output


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
seq_lengths = [seq.shape[0] for seq in inputs]

# Training loop
def train(model, num_epochs, lr, batch_size, X_train, y_train, seq_lengths):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size]
            targets = y_train[i:i+batch_size]
            lengths = seq_lengths[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(loss)
    return model

# initialize and train Vanilla RNN

# initialize and train Sequential NN fixing #timesteps to the minimum sequence length

# initialize and train Sequential NN fixing #timesteps to the maximum sequence length
# NOTE: it is OK to use torch.nn.utils.rnn.pad_sequence; make sure to set parameter batch_first correctly
