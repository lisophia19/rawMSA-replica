import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from preprocessing import *
import os
import numpy as np
import random

""""
model assumes input sequence is of length 31000 but we mgiht need to fix that...
embedding_dim = output_dim

tbd number of seqeunces + 200 residues max 
"""

class RSAProteinModel(nn.Module):
    # num_sequences corresponds to the number of vertical columns (sequences); equivalent to the Y-dimension
    def __init__(self, num_sequences):
        super(RSAProteinModel, self).__init__()

        # sequence_length=31, input_embedding_sz=26, embedding_dim=14, lstm_hidden=350, dropout_rate=0.4
        # Represents the number of residues per sequence; Equal to 31 for SS prediction
        self.sequence_length = 31

        # Represents number of tokens to embed
        self.input_embedding_sz = 26
        # Represents number of items for each vector
        self.embedding_dim = 14
        # For the LSTM BRNNs
        self.lstm_hidden = 350

        self.dropout_rate=0.4

        self.num_sequences = num_sequences

        #input_dim = 25 residues + pad, embed to 14
        self.embedding = nn.Embedding(num_embeddings=self.input_embedding_sz, embedding_dim=self.embedding_dim)

        #input channels = 14 and output channels = 14
        # Originally: padding = (0, 5)
        self.conv = nn.Conv2d(in_channels=self.embedding_dim, out_channels=self.embedding_dim, kernel_size=(1, 11), padding=(0, 5))
        self.pool = nn.MaxPool2d(kernel_size=(1, 10), stride=(1, 10))

        # LSTM input size = 14*2
        self.bilstm1 = nn.LSTM(input_size=(self.embedding_dim * (self.num_sequences // 10)), hidden_size=self.lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=2*self.lstm_hidden, hidden_size=self.lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True) #two bilstm of size 350

        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc1 = nn.Linear(self.sequence_length * self.lstm_hidden, 200)
        # self.dl1 = nn.Linear()
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 4) #9 secondary structures from preprocessing

    def forward(self, x):
        # x: (batch_size, 31*max_depth=31*15=465)  -- max depth = number of sequences
        x = self.embedding(x)  # -> (batch_size, 465, 14)        
        x = x.view(-1, 31, self.num_sequences, self.embedding_dim) #reshape to (batch_size, 31, 15, 14)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels=14, height=31, width=5)
        
        x = F.relu(self.conv(x)) # -> (batch, 14, 31, 15)
        x = self.pool(x)  # -> (batch_size, 14, 31, 2)

        x = x.permute(0, 2, 1, 3).contiguous()  # -> (batch, 31, 14, 2)
        x = x.view(x.size(0), 31, -1)  # (batch, 31, 28)

        lstm_out1, _ = self.bilstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)

        lstm_out = 0.5 * (lstm_out2[:, :, :lstm_out2.size(2)//2] + lstm_out2[:, :, lstm_out2.size(2)//2:])

        x = lstm_out.contiguous().view(x.size(0), -1)  # flatten
        x = self.dropout(F.linear(x, self.fc1.weight, self.fc1.bias))
        x = self.dropout(F.linear(x, self.fc2.weight, self.fc2.bias))
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)

    # labels are NOT one-hot, but indices
    def loss(self, predictions, labels):
        # predictions: (batch_size, num_classes)
        # labels: (batch_size, num_classes)
        # return F.cross_entropy(predictions, labels)
        return F.nll_loss(predictions, labels)

    def accuracy(self, predictions, labels):
        pred_classes = torch.argmax(predictions, dim=1)
        # true_classes = torch.argmax(labels, 1)
        # correct_prediction = torch.eq(pred_classes, true_classes)
        correct_prediction = (pred_classes == labels)
        # return torch.mean(torch.Tensor(correct_prediction).to(torch.float32))
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
        label = self.labels[:, idx]  # Need these to be of shape (31*Y) as well

        return window_flat, label, idx
   
def train_data_processing(train_body_seq_dict, train_master_seq_dict):
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
            # print(f"Number of Master Sequences for PID: {family_id}: ",  num_master_seqs)

            random_index = random.randrange(0, num_master_seqs)
            # print(random_index)

            master_seq = curr_master_seqs[0][random_index]
            master_seq_label = curr_master_seqs[1][random_index]

            batch_data = batch_train_data(train_body_seq_dict, batch_num, family_id)

            # curr_batch = [master_seq] + batch_data
            master_seq = torch.tensor([master_seq])
            curr_batch = torch.concat((master_seq, batch_data))

            training_inputs.append(curr_batch) #master seq is first in list
            # training_inputs += batch_data
            training_labels.append([master_seq_label]) #training labels only has one label aka master
            # print(type(master_seq))

    # print(np.array(training_inputs).shape)

    # print(np.array(training_labels).shape)

    training_inputs = torch.tensor(np.array(training_inputs)).permute(0, 2, 1)
    training_labels = torch.tensor(np.array(training_labels)).permute(0, 2, 1)

    return torch.tensor(training_inputs), torch.tensor(training_labels),


def batch_step(optimizer, model, item, is_training = True):
    optimizer.zero_grad()   # clear gradients beforehand

    input_tensor = item[0]  # corresponds to the input_tensor for the current sliding window
    labels_tensor = item[1] # Corresponds to the labels for ALL of the current sequences
    #curr_index = item[2]    # Corresponds to current index of the master sequence we are predicting for

    y_pred = model(input_tensor)   # Outputs (1,9) vector of softmax values

    # Get label at current idx for master sequence
    actual_label = labels_tensor.long() - 1  # make sure it's a LongTensor
    actual_label = actual_label.squeeze()           # match (batch_size=1,) if necessary

    # print("PREDICTIONS SHAPE: ", y_pred.shape)
    # print("LABELS SHAPE: ", actual_label.shape)

    loss = model.loss(y_pred, actual_label)

    if is_training:
        loss.backward()
        optimizer.step()

    # evaluation after epoch - accuracy computation

    return (loss.item(), model.accuracy(y_pred, actual_label))


def main():
    # Master Sequences: Dictionary of list (tuple of tensors)
    # Key: Family ID, Value: list
    # Pull a random Master sequence from the current domain

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_master_seq_dict = gather_master_sequences([])
    test_master_seq_dict = gather_master_sequences([], data_type="test")
    val_master_seq_dict = gather_master_sequences([], data_type='val')

    #build datasets using the custom sliding window class
    #train_dataset = MSASlidingWindowDataset(train_msa_tensor, train_labels_tensor, window_size=31, max_depth=15)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_body_seq_dict, test_body_seq_dict, val_body_seq_dict = gather_body_sequences()

    #load in data of size N x L;
    # N = number of sequences
    # L = length of each sequence

    # #create model
    model = RSAProteinModel(num_sequences=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

    # move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # training loop for 5 epochs (up to 5 in paper)
    epochs = 5
    for j in range(epochs):
        # Train Data
        train_sequences_tensor, train_labels_tensor = train_data_processing(train_body_seq_dict, train_master_seq_dict)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        num_train_entries = train_sequences_tensor.shape[0]

        train_dataset = MSASlidingWindowDataset(train_sequences_tensor, train_labels_tensor, window_size=31, max_depth=15)
        curr_loss = 0.
        train_acc = 0.

        # Val Data
        val_sequences_tensor, val_labels_tensor =  train_data_processing(val_body_seq_dict, val_master_seq_dict)
        val_dataset = MSASlidingWindowDataset(val_sequences_tensor, val_labels_tensor, window_size=31, max_depth=15) # Shape of (4800, 15, 31 --> for 200 times), 
        val_loss = 0.
        val_acc = 0.

        num_val_entries = val_sequences_tensor.shape[0]
        
        # Train for N-total batches of batch-size M (i.e. 15) for EACH family domain as well
        for item in tqdm(train_dataset, desc=f"Epoch {j+1} Training"):
            # print(item[1].shape)

            loss, acc = batch_step(optimizer, model, item)
            curr_loss += loss
            train_acc += acc
        
        # print(len(train_dataset))
        print(f"After epoch {j+1}: Accuracy ={train_acc / len(num_train_entries):.4f}; Running Loss = {curr_loss:.4f}")

        for item in tqdm(val_dataset):
            loss, acc = batch_step(optimizer, model, item, is_training=False)
            val_loss += loss
            val_acc += acc

        print(f"Validation Accuracy after epoch {j+1}: Accuracy = {val_acc / len(num_val_entries):.4f}; Running Loss = {val_loss:.4f}")
        print()

    # Test data
    test_sequences_tensor, test_labels_tensor =  train_data_processing(test_body_seq_dict, test_master_seq_dict)
    test_dataset = MSASlidingWindowDataset(test_sequences_tensor, test_labels_tensor, window_size=31, max_depth=15)
    test_loss = 0.
    test_acc = 0.

    num_test_entries = test_sequences_tensor.shape[0]


    for item in tqdm(test_dataset):
        loss, acc = batch_step(optimizer, model, item, is_training=False)
        test_loss += loss
        test_acc += acc

    print(f"Test Accuracy after epoch {j+1}: Accuracy = {test_acc / len(num_test_entries):.4f}; Running Loss = {test_loss:.4f}")

main()
