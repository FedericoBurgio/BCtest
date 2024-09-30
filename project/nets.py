import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(state_size, 256)
        
        # Hidden layers
   
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        
        # Output layer
        self.fc5 = nn.Linear(256, action_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)  # Reduced dropout rate back to 0.1
        
        # Activation functions
        self.relu = nn.ReLU()  # Switched back to standard ReLU

    def forward(self, state):
        # Input -> hidden layer 1
        x = self.relu(self.fc1(state))
        
        # # Hidden layer 1 -> hidden layer 2
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)  # Apply dropout after activation
        
        # Hidden layer 2 -> hidden layer 3
        x = self.relu(self.fc3(x))
        #x = self.dropout(x)  # Dropout for regularization
        
        # Hidden layer 3 -> hidden layer 4
        x = self.relu(self.fc4(x))
        #x = self.dropout(x)
        
        # Output layer
        action_logits = self.fc5(x)
        
        return action_logits
