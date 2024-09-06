import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Check if CUDA is available
lossHistory = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data = np.load("/home/atari_ws/project/DatasetBC.npz")
data = np.load("dataset20.npz")


# Extract states and actions
states = data['states']
actions = data['actions']

state_size = states.shape[1]  
action_size = actions.shape[1]  

valid_mask = ~np.isnan(actions).any(axis=1) & ~np.isnan(states).any(axis=1)

# Filter out states and actions with NaN values
states = states[valid_mask]
actions = actions[valid_mask]

# Convert to PyTorch tensors for training
states = torch.tensor(states, dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.float32).to(device)  # or long, depending on action type

dataset = TensorDataset(states, actions)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define a simple neural network policy
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits

# # Assuming state_size and action_size can be determined from the data
# state_size = states.shape[1]  # The number of features in each state
# action_size = torch.max(actions)  # Assuming actions are labeled from 0 to (max action)

# Initialize the policy network
model = PolicyNetwork(state_size, action_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Use CrossEntropyLoss for discrete actions
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000


for epoch in range(num_epochs):
    for state_batch, action_batch in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(state_batch)
        
        # Compute loss
        loss = criterion(predictions, action_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    lossHistory.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

np.save("lossHistory.npy", lossHistory)
# Save the trained model
model_path = "policy_model22.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

