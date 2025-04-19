import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

""""
model assumes input sequence is of length 31000 but we mgiht need to fix that...
embedding_dim = output_dim
"""
class RSAProteinModel(nn.Module):
    def __init__(self, input_dim=26, embedding_dim=14, lstm_hidden=350, dropout_rate=0.4):
        super(RSAProteinModel, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)

        self.conv = nn.Conv2d(in_channels=1, out_channels=embedding_dim, kernel_size=(1, 10), padding=(0, 5))
        self.pool = nn.MaxPool2d(kernel_size=(1, 20), stride=(1, 20))

        self.bilstm1 = nn.LSTM(input_size=700, hidden_size=lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=lstm_hidden, hidden_size=lstm_hidden, num_layers=1, 
                               batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(31 * lstm_hidden, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 3)

    def forward(self, x):
        # x shape: (batch_size, 31000)
        x = self.embedding(x)  # -> (batch_size, 31000, 14)
        x = x.view(-1, 31, 1000, 14)
        x = x.permute(0, 3, 1, 2)  # (batch, channels=14, height=31, width=1000)
        x = F.relu(self.conv(x))
        x = self.pool(x)  # -> (batch, channels=14, height=31, width=50)

        x = x.permute(0, 2, 1, 3).contiguous()  # -> (batch, 31, 14, 50)
        x = x.view(x.size(0), 31, -1)  # -> (batch, 31, 700)

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

    def loss(self, predictions, labels):
        nll_comps = -labels * torch.log(torch.clip(predictions,1e-10,1.0))
        return torch.mean(torch.sum(nll_comps, axis=[1]))

    def accuracy(self, predictions, labels):
        pred_classes = torch.argmax(predictions, 1)
        true_classes = torch.argmax(labels, 1)
        correct_prediction = torch.eq(pred_classes, true_classes)
        return torch.mean(torch.Tensor(correct_prediction).to(torch.float32))
        
class MSASlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, msa_tensor, labels, window_size=31, max_depth=1000, pad_token=0):
        """
        msa_tensor: shape (L, Y) - raw integer-encoded MSA (1-25 for residues, 0 for padding)
        labels: shape (L, num_classes) - one-hot vectors per residue
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
        msa_padded[pad_len:pad_len+L, :Y] = msa_trimmed

        self.msa_padded = msa_padded
        self.length = L

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.msa_padded[idx:idx+self.window_size, :]  # (31, Y)
        window_flat = window.flatten()  # (31*Y,)
        label = self.labels[idx]  # one-hot (num_classes,)
        return window_flat, label
   

def main():
    #load in data
    # Example mock inputs (you can replace with real MSA/label tensors)
    # Suppose MSA = (L, Y) and labels = (L, num_classes)
    L, Y = 200, 500  # e.g., 200 residues, 500 aligned sequences
    num_classes = 4

    # MSA DATA RANDOM RN NEED TO CHANGE
    train_msa = torch.randint(1, 26, (L, Y))
    train_labels = F.one_hot(torch.randint(0, num_classes, (L,)), num_classes=num_classes).float()

    test_msa = torch.randint(1, 26, (L, Y))
    test_labels = F.one_hot(torch.randint(0, num_classes, (L,)), num_classes=num_classes).float()

    # Build datasets using the custom sliding window class
    train_dataset = MSASlidingWindowDataset(train_msa, train_labels)
    test_dataset = MSASlidingWindowDataset(test_msa, test_labels)


    # dataloaders are an easy way to batch and shuffle datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True) #IMPORTANT!!!
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    #create model
    model = RSAProteinModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # training loop for 5 epochs (up to 5 in paper)
    epochs = 5

    for j in range(epochs):
        model.train()
        for batch_idx, (input, label) in tqdm(enumerate(train_loader), desc=f"Epoch {j+1} Training"):
            optimizer.zero_grad()
            
            # move data to device
            input, label = input.to(device), label.to(device)

            y_pred = model(input)  # forward pass
            loss = model.loss(y_pred, label)  # compute loss
            loss.backward()  # backprop
            optimizer.step()  # update weights

        # evaluation after epoch
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label.to(device)
                test_acc += model.accuracy(model(input), label)

        print(f"Accuracy on testing set after epoch {j+1}: {test_acc / len(test_loader):.4f}")

    print()
    print(model)
