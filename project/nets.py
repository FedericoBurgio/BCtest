import torch
import torch.nn as nn
import torch.nn.functional as F

#ERA QUELLA BUONA
class PolicyNetworkBUONO(nn.Module):# LSTM, 256, 256
    def __init__(self, input_size, output_size):
        hidden_layers = [256,256]
        rnn_hidden_size = 256
        num_rnn_layers = 1
        super(PolicyNetworkBUONO, self).__init__()
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        #self.attention = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4)
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

# class PolicyNetworkBUONO2(nn.Module):
#     def __init__(self, input_size, output_size, hidden_layers, rnn_hidden_size, num_rnn_layers, num_attention_heads=4):
#         super(PolicyNetworkBUONO2, self).__init__()
        
#         # GRU Layer
#         self.rnn = nn.GRU(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        
#         # Multi-Head Attention Layer
#         self.attention = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=num_attention_heads, batch_first=True)
        
#         # Fully Connected Layers
#         self.fc_layers = nn.ModuleList()
#         self.fc_layers.append(nn.Linear(rnn_hidden_size, hidden_layers[0]))
#         self.fc_layers.append(nn.ReLU())
#         for i in range(1, len(hidden_layers)):
#             self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
#             self.fc_layers.append(nn.ReLU())
        
#         # Final Output Layer
#         self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
#     def forward(self, x):
#         # GRU Forward Pass
#         gru_out, _ = self.rnn(x)  # gru_out shape: [batch_size, sequence_length, rnn_hidden_size]
        
#         # Multi-Head Attention Mechanism
#         # The attention layer expects (sequence_length, batch_size, embedding_size), so we need to permute
#         attention_out, _ = self.attention(gru_out, gru_out, gru_out)  # Self-attention over GRU output
        
#         # Use the output of the attention layer from the last time step
#         x = attention_out[:, -1, :]  # Take the last time step for downstream
        
#         # Pass through fully connected layers
#         for layer in self.fc_layers:
#             x = layer(x)
        
#         return x

class PolicyNetworkBUONO3(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetworkBUONO3, self).__init__()
        
        hidden_layers = [256]
        transformer_hidden_size = 256
        num_transformer_layers = 2
        num_attention_heads = 4
        
        # Linear layer to project input to transformer hidden size
        self.input_projection = nn.Linear(input_size, transformer_hidden_size)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_hidden_size, nhead=num_attention_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Fully Connected Layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(transformer_hidden_size, hidden_layers[0]))
        self.fc_layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.fc_layers.append(nn.ReLU())
        
        # Final Output Layer
        self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))
        
    def forward(self, x):
        # Project input to transformer hidden size
        x = self.input_projection(x)
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)
        
        # Use the output of the last time step
        x = transformer_out[:, -1, :]  # Take the last time step for downstream
        
        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x

class PolicyNetwork(nn.Module):#vecchio?
    def __init__(self, input_size, output_size, 
                 ):
        super(PolicyNetwork, self).__init__()
        hidden_layers = [256,256]
        rnn_hidden_size = 256 
        num_rnn_layers = 1
        
        # Linear layer to project 82 to 256
        self.input_projection = nn.Linear(input_size, rnn_hidden_size)
        
        # Transformer Encoder Layer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=rnn_hidden_size, nhead=4), num_layers=2
        )
        
        # LSTM Layer
        self.rnn = nn.GRU(rnn_hidden_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
        
        # Fully Connected Layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(rnn_hidden_size, hidden_layers[0]))
        self.fc_layers.append(nn.ReLU())
        for i in range(1, len(hidden_layers)):
            self.fc_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.fc_layers.append(nn.ReLU())
        
        # Output Layer
        self.fc_layers.append(nn.Linear(hidden_layers[-1], output_size))    
        
    def forward(self, x):
        # Project input to match transformer dimension
        x = self.input_projection(x)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # LSTM
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Get the output from the last time step
        
        # Fully Connected Layers
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

class PolicyNetworkLSTM(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size = 256  
        num_layers = 1
        super(PolicyNetworkLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Initialize hidden state and cell state
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM output
        lstm_out, _ = self.lstm(x, (h_0, c_0))  # lstm_out shape: (batch_size, seq_length, hidden_size)

        # Apply attention mechanism
        # Query, Key, and Value are all lstm_out in this case
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)  # attn_output shape: (batch_size, seq_length, hidden_size)

        # Take the output corresponding to the last time step
        out = self.fc(attn_output[:, -1, :])  # out shape: (batch_size, output_size)

        return out

class ThreeLayerDenseNet(nn.Module):
    def __init__(self, input_size, output_size):
        hidden_size = 256
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