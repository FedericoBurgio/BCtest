import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os
import h5py
from collections import deque

# Define your PolicyNetwork class
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[256], rnn_hidden_size=256, num_rnn_layers=2):
        super(PolicyNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, num_layers=num_rnn_layers, batch_first=True)
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

# Get the current date and time
now = datetime.now()
date_time_str = now.strftime("%d%H%M")  # Format as DAY-HOUR-MINUTE

# Create the directory if it doesn't exist
save_dir = "models/" + f"{date_time_str}"
os.makedirs(save_dir, exist_ok=True)

# Load the dataset
datasetName = "boi000.h5"  # Update the file name as needed
with h5py.File("datasets/" + datasetName, 'r') as f:
    states = f['states'][:]  # Read all data from the 'states' dataset
    qNext = f['qNext'][:]    # Read all data from the 'qNext' dataset
print("Number of samples:", len(states))

# Remove NaN and Inf values
valid_mask = ~np.isnan(qNext).any(axis=1) & ~np.isnan(states).any(axis=1)
valid_mask &= ~np.isinf(qNext).any(axis=1) & ~np.isinf(states).any(axis=1)
states = states[valid_mask]
qNext = qNext[valid_mask]

# Optionally downsample the data
step = 1
states = states[::step]
qNext = qNext[::step]

# Define your desired sequence length
sequence_length = 3  # You can adjust this value as needed

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states = torch.tensor(states, dtype=torch.float32).to(device)
actions = torch.tensor(qNext, dtype=torch.float32).to(device)

# Create overlapping sequences
num_sequences = states.shape[0] - sequence_length + 1
states_seq = states.unfold(0, sequence_length, 1)  # Shape: [num_sequences, sequence_length, state_size]
actions_seq = actions[sequence_length - 1:]        # Actions correspond to the last time step in each sequence

# Ensure sequences are contiguous in memory
states_seq = states_seq.contiguous()
actions_seq = actions_seq.contiguous()

# Create datasets and dataloaders
dataset = TensorDataset(states_seq, actions_seq)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

# Define model parameters
state_size = states_seq.shape[2]   # Input size is the last dimension of states_seq
action_size = actions_seq.shape[1]
hidden_layers = [256, 256]
rnn_hidden_size = 256
num_rnn_layers = 1

# Initialize the model
model = PolicyNetwork(state_size, action_size, hidden_layers, rnn_hidden_size, num_rnn_layers).to(device)
np.savez(save_dir + "/NNprop.npz", s_size=state_size, a_size=action_size, h_layers=hidden_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)

# Training loop variables
num_epochs = 250  # Adjust the number of epochs as needed
best_val_loss = float('inf')
best_policy = None
train_losses = []
val_losses = []

# Training and validation loop
try:
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predicted_train_actions = model(batch_states)
            train_loss = criterion(predicted_train_actions, batch_actions)
            
            # Backward pass and optimization
            train_loss.backward()
            optimizer.step()
            total_train_loss += train_loss.item()
        average_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                predicted_val_actions = model(batch_states)
                val_loss = criterion(predicted_val_actions, batch_actions)
                total_val_loss += val_loss.item()
        average_val_loss = total_val_loss / len(val_loader)

        scheduler.step(average_val_loss)

        # Store losses
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        # Save best model based on validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_policy = model.state_dict()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4e}, '
              f'Val Loss: {average_val_loss:.4e}, Best Val Loss: {best_val_loss:.4e}')
        
        # Save the best policy every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = f'{save_dir}/best_policy_ep{epoch+1}.pth'
            torch.save(best_policy, model_path)
            print(f'Model saved to {model_path}')
        
except KeyboardInterrupt:
    print("Training interrupted. Saving the best model...")

# Save the final best model
model_path = f"{save_dir}/best_policy_epfinal.pth"
torch.save(best_policy, model_path)
print(f"Final best model saved to {model_path}")

# Save loss history as numpy arrays
np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
print(f"Loss history saved to {save_dir}/train_losses.npy and val_losses.npy")
