import torch
import torch.nn as nn
import torch.nn.functional as F

# class PolicyNetwork__(nn.Module):# LTMS, attention, 256, 256
#     def __init__(self, input_size, output_size, hidden_layers=[256, 256], rnn_hidden_size=256, num_rnn_layers=1):
#         super(PolicyNetwork, self).__init__()
        
#         # LSTM layer
#         self.rnn = nn.LSTM(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        
#         # Attention Layer: Weights to learn attention over LSTM outputs
#         self.attention_weights = nn.Linear(rnn_hidden_size, 1)
        
#         # Fully connected layers
#         self.fc_layers = nn.ModuleList()
        
#         # First hidden layer
#         self.fc_layers.append(nn.Linear(rnn_hidden_size, hidden_layers[0]))
#         self.fc_layers.append(nn.ReLU())
        
#         # Additional hidden layers
#         for i in range(1, len(hidden_layers)):
#             self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
#             self.fc_layers.append(nn.ReLU())
        
#         # Final output layer
#         self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
#     def forward(self, x):
#         # Forward pass through the LSTM
#         lstm_out, _ = self.rnn(x)  # lstm_out shape: (batch_size, sequence_length, rnn_hidden_size)
        
#         # Compute attention scores for each time step
#         attention_scores = self.attention_weights(lstm_out)  # Shape: (batch_size, sequence_length, 1)
#         attention_scores = attention_scores.squeeze(-1)  # Shape: (batch_size, sequence_length)
        
#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (batch_size, sequence_length)
        
#         # Compute the weighted sum of LSTM outputs (context vector)
#         context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)  # Shape: (batch_size, rnn_hidden_size)
        
#         # Pass the context vector through the fully connected layers
#         x = context_vector
#         for layer in self.fc_layers:
#             x = layer(x)
        
#         return x

class PolicyNetwork(nn.Module):# LTMS, 256, 256
    def __init__(self, input_size, output_size, hidden_layers=[256], rnn_hidden_size=256, num_rnn_layers=2):
        super(PolicyNetwork, self).__init__()
        self.rnn = nn.GRU(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        self.fc_layers = nn.ModuleList()
        
        # First hidden layer
        self.fc_layers.append(nn.Linear(rnn_hidden_size, hidden_layers[0]))
        self.fc_layers.append(nn.ReLU())
        
        # Additional hidden layers
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.fc_layers.append(nn.ReLU())
        
        # Final output layer
        self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
    def forward(self, x):
        x, _ = self.rnn(x)  # Forward pass through RNN
        x = x[:, -1, :]  # Get the output from the last time step
        for layer in self.fc_layers:
            x = layer(x)
        return x


class PolicyNetwork__(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=2, hidden_layers=[256, 256]):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Attention mechanism
        self.attention_layer = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        # First hidden layer
        self.fc_layers.append(nn.Linear(hidden_size, hidden_layers[0]))
        self.fc_layers.append(nn.ReLU())
        # Additional hidden layers
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.fc_layers.append(nn.ReLU())
        # Final output layer
        self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [batch_size, seq_len, hidden_size]
        
        # Attention mechanism
        # Compute attention scores
        attention_scores = self.attention_layer(lstm_out)  # Shape: [batch_size, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Shape: [batch_size, seq_len, 1]
        # Compute context vector as weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: [batch_size, hidden_size]
        
        # Pass through fully connected layers
        x = context_vector
        for layer in self.fc_layers:
            x = layer(x)
        return x


class ThreeLayerDenseNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ThreeLayerDenseNet, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # No activation on the output layer (e.g., for regression)
        return x