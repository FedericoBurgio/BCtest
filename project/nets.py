import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[256,256], rnn_hidden_size=256, num_rnn_layers=1):
        super(PolicyNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        self.fc_layers = nn.ModuleList()
        
        # First hidden layer
        self.fc_layers.append(nn.Linear(rnn_hidden_size, hidden_layers[0]))
        self.fc_layers.append(nn.ReLU())
        
        # Additional hidden layers
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.fc_layers.append(nn.ReLU())
        
        # Final output layer
        self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
    def forward(self, x):
        x, _ = self.rnn(x)  # Forward pass through RNN
        x = x[:, -1, :]  # Get the output from the last time step
        for layer in self.fc_layers:
            x = layer(x)
        return x
