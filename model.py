import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from preprocessing import map_to_integer
import os

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

        self.fc1 = nn.Linear(self.sequence_length * self.lstm_hidden, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 9) #9 secondary structures from preprocessing

    def forward(self, x):
        # x: (batch_size, 31*max_depth=31*25=775)  -- max depth = number of sequences
        x = self.embedding(x)  # -> (batch_size, 465, 14)

        # print("NUM OF SEQUNECES: ", x.shape)
        
        x = x.view(-1, 31, self.num_sequences, self.embedding_dim) #reshape to (batch_size, 31, 25, 14)
        x = x.permute(0, 3, 1, 2)  # (batch_size, channels=14, height=31, width=25)
        
        x = F.relu(self.conv(x)) # -> (batch, 14, 31, 25)
        x = self.pool(x)  # -> (batch_size, 14, 31, 2)

        x = x.permute(0, 2, 1, 3).contiguous()  # -> (batch, 31, 14, 2)
        x = x.view(x.size(0), 31, -1)  # (batch, 31, 28)

        # print("SHAPE AFTER POOLING: ", x.shape)

        lstm_out1, _ = self.bilstm1(x)
        lstm_out1 = self.dropout(lstm_out1)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        lstm_out2 = self.dropout(lstm_out2)

        lstm_out = 0.5 * (lstm_out2[:, :, :lstm_out2.size(2)//2] + lstm_out2[:, :, lstm_out2.size(2)//2:])

        x = lstm_out.contiguous().view(x.size(0), -1)  # flatten
        x = self.dropout(F.linear(x, self.fc1.weight, self.fc1.bias))
        x = self.dropout(F.linear(x, self.fc2.weight, self.fc2.bias))
        x = self.fc3(x)

        # Removed the log_softmax
        return F.log_softmax(x, dim=-1)

    def loss(self, predictions, labels):
        # predictions: (batch_size, num_classes)
        # labels: (batch_size, num_classes)
        return F.nll_loss(predictions, torch.argmax(labels, dim=1))

    def accuracy(self, predictions, labels):
        pred_classes = torch.argmax(predictions, 1)
        true_classes = torch.argmax(labels, 1)
        correct_prediction = torch.eq(pred_classes, true_classes)
        return torch.mean(torch.Tensor(correct_prediction).to(torch.float32))
        
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
        L, Y = msa_tensor.shape
        Y = min(Y, max_depth)

        msa_trimmed = msa_tensor[:, :Y]

        msa_padded = torch.full((L + 2 * pad_len, max_depth), pad_token, dtype=torch.long)
        msa_padded[pad_len:pad_len+L, :Y] = torch.tensor(msa_trimmed, dtype=torch.long)

        # print("MSA PADDED SHAPE: ", msa_padded.shape)

        self.msa_padded = msa_padded
        self.length = L


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.msa_padded[idx:idx+self.window_size, :]  # (31, Y)
        window_flat = window.flatten()  # (31*Y,)
        label = self.labels[idx]  # Need these to be of shape (31*Y) as well

        return window_flat, label, idx
   

def main():
    #load in data of size N x L;
    # N = number of sequences
    # L = length of each sequence
    # train_msa_tensor, train_labels_tensor = map_to_integer(Path.cwd() / "collected_data" / "data.txt")
    train_msa_tensor, train_labels_tensor = map_to_integer(os.path.join("collected_data", "PF00071.parsed.txt"))
    train_msa_tensor = torch.from_numpy(train_msa_tensor.T).long()
    train_labels_tensor = torch.from_numpy(train_labels_tensor.T)

    print(train_labels_tensor.shape)

    # Build datasets using the custom sliding window class
    train_dataset = MSASlidingWindowDataset(train_msa_tensor, train_labels_tensor, max_depth=train_msa_tensor.shape[1])

    #test_dataset = MSASlidingWindowDataset(test_msa, test_labels)

    # dataloaders are an easy way to batch and shuffle datasets
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) #what should batch size be 
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    #create model
    model = RSAProteinModel(num_sequences=train_msa_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # training loop for 5 epochs (up to 5 in paper)
    epochs = 5
    for j in range(epochs):
        model.train()
        for item in tqdm(train_dataset, desc=f"Epoch {j+1} Training"):
            optimizer.zero_grad()   # clear gradients beforehand

            input_tensor = item[0]  # corresponds to the input_tensor for the current sliding window
            labels_tensor = item[1] # Corresponds to the labels for ALL of the current sequences
            curr_index = item[2]    # Corresponds to current index of the master sequence we are predicting for


            y_pred = model(input_tensor)   # Outputs (1,9) vector of softmax values

            # Get label at current idx for master sequence
            actual_label = int(labels_tensor[0].item())
            # print("ACTUAL LABEL: ",actual_label)
            one_hot_actual = torch.zeros((1, 9))
            one_hot_actual[:, actual_label-1] = 1

            loss = model.loss(y_pred, one_hot_actual)
            loss.backward()
            optimizer.step()
            # evaluation after epoch

        model.eval()
        test_acc = 0
        with torch.no_grad():
            for item in train_dataset:
                # input, label = input.to(device), label.to(device)
                input_tensor = item[0]  # corresponds to the input_tensor for the current sliding window
                labels_tensor = item[1] # Corresponds to the labels for ALL of the current sequences
                curr_index = item[2]    # Corresponds to current index of the master sequence we are predicting for


                y_pred = model(input_tensor)   # Outputs (1,9) vector of softmax values

                # Get label at current idx for master sequence
                actual_label = int(labels_tensor[0].item())
                # print("ACTUAL LABEL: ",actual_label)
                one_hot_actual = torch.zeros((1, 9))
                one_hot_actual[:, actual_label-1] = 1

                test_acc += model.accuracy(y_pred, one_hot_actual)

        print(f"Accuracy on testing set after epoch {j+1}: {test_acc / len(train_dataset):.4f}")

    # for j in range(epochs):
    #     model.train()
    #     for batch_idx, (input, label, idx) in tqdm(enumerate(train_loader), desc=f"Epoch {j+1} Training"):
    #     # for input, label in tqdm(train_dataset, desc=f"Epoch {j+1} Training"):
    #         optimizer.zero_grad()   # Clear gradients before iteration
            
            # move data to device
            #input, label = input.to(device), label.to(device)
            
            # y_pred = model(input)  # forward pass ==> softmax values
            # print(label.shape)
            # print(idx.shape)

            # loss = model.loss(y_pred, label)  # compute loss
            # loss.backward()  # backprop
            # optimizer.step()  # update weights

        # evaluation after epoch
        # model.eval()
        # test_acc = 0
        # with torch.no_grad():
        #     for batch_idx, (input, label) in enumerate(test_loader):
        #         input, label = input.to(device), label.to(device)
        #         test_acc += model.accuracy(model(input), label)

        # print(f"Accuracy on testing set after epoch {j+1}: {test_acc / len(test_loader):.4f}")



    print()
    print(model)


main()