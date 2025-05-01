import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from preprocessing import *
import os
import numpy as np
import random

class RSAProteinModel(nn.Module):
    # num_sequences corresponds to the number of vertical columns (sequences); equivalent to the Y-dimension
    def __init__(self, num_sequences):
        super(RSAProteinModel, self).__init__()

        self.sequence_length = 31           # represents the number of residues per sequence; Equal to 31 for SS prediction
        self.input_embedding_sz = 25        # represents number of tokens to embed (corresponds to the 25 amino acids)
        self.embedding_dim = 14             # represents number of items for each vector (dimensionality of the embeddings)
        
        # For the LSTM BRNNs
        self.lstm_hidden = 350
        self.dropout_rate=0.25
        self.num_sequences = num_sequences


        # Below defines the layers for the model
        self.embedding = nn.Embedding(num_embeddings=self.input_embedding_sz, embedding_dim=self.embedding_dim)
        self.conv = nn.Conv2d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=(1, 11), padding=(0, 5))
        self.pool = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 10))

        self.bilstm1 = nn.LSTM(input_size=(self.embedding_dim * (self.num_sequences // 10)), hidden_size=self.lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=2*self.lstm_hidden, hidden_size=self.lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc1 = nn.Linear(self.sequence_length * self.lstm_hidden, 350)
        self.fc2 = nn.Linear(350, 100)
        self.fc3 = nn.Linear(100, 4) #4 secondary structures from preprocessing

    def forward(self, x, is_training):
        # x: (batch_size, 31*max_depth=31*15=465)  -- max depth = number of sequences
        x = self.embedding(x)  # -> (batch_size, 465, 14)        
        x = x.view(-1, 31, self.num_sequences, self.embedding_dim) #reshape to (batch_size, 31, 15, 14)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels=14, height=31, width=5)
        
        x = F.relu(self.conv(x)) # -> (batch, 14, 31, 15)
        x = self.pool(x)  # -> (batch_size, 14, 31, 2)

        x = x.permute(0, 2, 1, 3).contiguous()  # -> (batch, 31, 14, 2)
        x = x.view(x.size(0), 31, -1)  # (batch, 31, 28)

        lstm_out1, _ = self.bilstm1(x)
        if is_training:
            lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        if is_training:
            lstm_out2 = self.dropout(lstm_out2)

        lstm_out = 0.5 * (lstm_out2[:, :, :lstm_out2.size(2)//2] + lstm_out2[:, :, lstm_out2.size(2)//2:])

        x = lstm_out.contiguous().view(x.size(0), -1)  # flatten

        if is_training:
            x = self.dropout(F.linear(x, self.fc1.weight, self.fc1.bias))
            x = self.dropout(F.linear(x, self.fc2.weight, self.fc2.bias))
        else:
            x = F.linear(x, self.fc1.weight, self.fc1.bias)
            x = F.linear(x, self.fc2.weight, self.fc2.bias)
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)

    # labels are NOT one-hot, but indices
    def loss(self, predictions, labels):
        return F.nll_loss(predictions, labels)

    def accuracy(self, predictions, labels):
        pred_classes = torch.argmax(predictions, dim=1)
        correct_prediction = (pred_classes == labels)
        return torch.mean(correct_prediction.float())


class MSASlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, msa_tensor, labels, window_size=31, max_depth=25, pad_token=0):
        """
        msa_tensor: shape (L, Y) - raw integer-encoded MSA (1-25 for residues, 0 for padding)
        labels: shape (L, num_classes) - one-hot vectors per residue
        max_depth: number of sequences
        """
        super().__init__()
        self.window_size = window_size
        self.max_depth = max_depth
        self.pad_token = pad_token
        self.labels = labels

        pad_len = window_size // 2
        B, L, Y = msa_tensor.shape
        Y = min(Y, max_depth)

        msa_trimmed = msa_tensor[:, :, :Y]

        msa_padded = torch.full((B, L + 2 * pad_len, max_depth), pad_token, dtype=torch.long)
        msa_padded[:, pad_len:pad_len+L, :Y] = torch.tensor(msa_trimmed, dtype=torch.long)

        self.msa_padded = msa_padded
        self.length = L


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.msa_padded[:, idx:idx+self.window_size, :]  # (31, Y)
        window_flat = torch.flatten(window, start_dim=1, end_dim=2)  # (31*Y,)
        label = self.labels[:, idx]  # need these to be of shape (31*Y) as well

        return window_flat, label, idx
   
def train_data_processing(train_body_seq_dict, train_master_seq_dict, device):
    training_inputs = []
    training_labels = []

    shuffled_family_ids = list(train_body_seq_dict.keys())
    random.shuffle(shuffled_family_ids)

    for family_id in shuffled_family_ids:
        #shuffle sequence data
        random.shuffle(train_body_seq_dict[family_id])
        total_seq_len = len(train_body_seq_dict[family_id])
        num_batches = total_seq_len // 14

        for batch_num in range(num_batches):
            curr_master_seqs = train_master_seq_dict[family_id]
            num_master_seqs = len(curr_master_seqs[0])

            random_index = random.randrange(0, num_master_seqs)
            master_seq = curr_master_seqs[0][random_index]
            master_seq_label = curr_master_seqs[1][random_index]

            batch_data = batch_train_data(train_body_seq_dict, batch_num, family_id)

            master_seq = torch.tensor(master_seq, device=device)
            curr_batch = torch.concat((master_seq.unsqueeze(0), batch_data))

            master_seq_label = torch.tensor(master_seq_label, device=device)

            training_inputs.append(curr_batch) # master seq is first in list
            training_labels.append(master_seq_label.unsqueeze(0)) # training labels only has one label a.k.a. master

    training_inputs = torch.stack(training_inputs).permute(0, 2, 1)
    training_labels = torch.stack(training_labels).permute(0, 2, 1)

    return torch.tensor(training_inputs), torch.tensor(training_labels),


def batch_step(optimizer, model, item, device, is_training = True):
    optimizer.zero_grad()   # clear gradients beforehand

    input_tensor = move_data_to_device(item[0], device) # corresponds to the input_tensor for the current sliding window
    labels_tensor = move_data_to_device(item[1], device) # corresponds to the labels for ALL of the current sequences

    y_pred = model(input_tensor)   # outputs (1, 4) vector of softmax values

    actual_label = labels_tensor.long() - 1
    actual_label = actual_label.squeeze()

    loss = model.loss(y_pred, actual_label)

    if is_training:
        loss.backward()
        optimizer.step()
    
    return (loss.item(), model.accuracy(y_pred, actual_label))


def move_data_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    else:
        return data


def main():
    #connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Master Sequences: Dictionary of list (tuple of tensors)
    # Key: Family ID, Value: list
    # Pull a random Master sequence from the current domain
    train_master_seq_dict = move_data_to_device(gather_master_sequences([]), device)
    test_master_seq_dict = move_data_to_device(gather_master_sequences([], data_type="test"), device)
    val_master_seq_dict = move_data_to_device(gather_master_sequences([], data_type='val'), device)

    train_body_seq_dict, test_body_seq_dict, val_body_seq_dict = gather_body_sequences()
    train_body_seq_dict = move_data_to_device(train_body_seq_dict, device)
    test_body_seq_dict = move_data_to_device(test_body_seq_dict, device)
    val_body_seq_dict = move_data_to_device(val_body_seq_dict, device)

    #load in data of size N x L;
    # N = number of sequences
    # L = length of each sequence

    # create model
    model = RSAProteinModel(num_sequences=15)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # training loop for 5 epochs (up to 5 in paper)
    epochs = 10
    for j in range(epochs):
        # Train Data
        train_sequences_tensor, train_labels_tensor = train_data_processing(train_body_seq_dict, train_master_seq_dict, device=device)
        train_sequences_tensor = train_sequences_tensor.to(device)
        train_labels_tensor = train_labels_tensor.to(device)
	
        num_train_entries = train_sequences_tensor.shape[1]

        train_dataset = MSASlidingWindowDataset(train_sequences_tensor, train_labels_tensor, window_size=31, max_depth=15)
        curr_loss = 0.
        train_acc = 0.

        # Val Data
        val_sequences_tensor, val_labels_tensor =  train_data_processing(val_body_seq_dict, val_master_seq_dict, device=device)
        val_sequences_tensor = val_sequences_tensor.to(device)
        val_labels_tensor = val_labels_tensor.to(device)
        val_dataset = MSASlidingWindowDataset(val_sequences_tensor, val_labels_tensor, window_size=31, max_depth=15) # Shape of (4800, 15, 31 --> for 200 times), 
        val_loss = 0.
        val_acc = 0.

        num_val_entries = val_sequences_tensor.shape[1]
        
        count = 0
        for item in tqdm(train_dataset, desc=f"Epoch {j+1} Training"):
            count+=1
            loss, acc = batch_step(optimizer, model, item, device)
            curr_loss += loss
            train_acc += acc
        
        print(f"After epoch {j+1}: Accuracy ={train_acc / num_train_entries:.4f}; Running Loss = {curr_loss:.4f}")

        for item in tqdm(val_dataset):
            loss, acc = batch_step(optimizer, model, item, device, is_training=False)
            val_loss += loss
            val_acc += acc

        print(f"Validation Accuracy after epoch {j+1}: Accuracy = {val_acc / num_val_entries:.4f}; Running Loss = {val_loss:.4f}")
        print()

    # Test data
    test_sequences_tensor, test_labels_tensor =  train_data_processing(test_body_seq_dict, test_master_seq_dict, device=device)
    test_sequences_tensor = test_sequences_tensor.to(device)
    test_labels_tensor = test_labels_tensor.to(device)
    test_dataset = MSASlidingWindowDataset(test_sequences_tensor, test_labels_tensor, window_size=31, max_depth=15)
    test_loss = 0.
    test_acc = 0.

    num_test_entries = test_sequences_tensor.shape[1]

    for item in tqdm(test_dataset):
        loss, acc = batch_step(optimizer, model, item, device, is_training=False)
        test_loss += loss
        test_acc += acc

    print(f"Test Accuracy after epoch {j+1}: Accuracy = {test_acc / num_test_entries:.4f}; Running Loss = {test_loss:.4f}")

main()
