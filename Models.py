import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length, output_size, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        # Calculate the size of the output after conv and pooling layers
        conv_output_size = hidden_size * 2 * (sequence_length // 4)  # Assuming two max pooling operations with stride=2

        self.fc = nn.Linear(conv_output_size, output_size)

    def forward(self, input_seq):
        x = input_seq.transpose(1, 2)  # Switch to (batch_size, features, sequence_length) for Conv1d
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, h_n = self.rnn(input_seq, h_0)
        output = self.dropout(output)
        output = self.linear(output[:, -1, :])
        return output

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    def forward(self, input_seq):
        h_0 = torch.randn(self.num_layers, input_seq.shape[0], self.hidden_size).to(self.device)
        output, h_n = self.gru(input_seq, h_0)
        output = self.dropout(output)
        output = self.linear(output[:, -1, :])
        return output


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        output = self.dropout(output)
        output = self.linear(output[:, -1, :])
        return output
    
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, device="cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        x = input_seq.transpose(1, 2)  # Switch to (batch_size, features, sequence_length) for Conv1d
        x = self.conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)  # Switch back to (batch_size, sequence_length, features) for LSTM

        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        output, (h, c) = self.lstm(x, (h_0, c_0))
        output = self.dropout(output)
        output = self.linear(output[:, -1, :])
        return output
    
