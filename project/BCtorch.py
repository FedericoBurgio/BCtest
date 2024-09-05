import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset
with open("/home/atari_ws/project/DatasetBC.pkl", 'rb') as f:
    data = pickle.load(f)

print(data[1]['s'])
print(len(data[1]['s']))

# Extract states and actions from the dataset
states = [entry['s'] for entry in data]
actions = [entry['a'] for entry in data]

# Convert states and actions to numpy arrays
states = np.array(states)
actions = np.array(actions)

print(states[1])
print(len(states[1]))

# Convert numpy arrays to PyTorch tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)

# Move tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states = states.to(device)
actions = actions.to(device)

# Define the neural network model
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Instantiate the model
input_dim = states.shape[1]
output_dim = actions.shape[1]
model = BCModel(input_dim, output_dim)

# Move the model to GPU if available
model = model.to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# Train the model
num_epochs = 1
batch_size = 512
validation_split = 0.2

# Split the data into training and validation sets
val_size = int(len(states) * validation_split)
train_states, val_states = states[:-val_size], states[-val_size:]
train_actions, val_actions = actions[:-val_size], actions[-val_size:]

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    # Shuffle the training data
    perm = torch.randperm(train_states.size(0))
    train_states, train_actions = train_states[perm], train_actions[perm]

    # Mini-batch training
    for i in range(0, train_states.size(0), batch_size):
        batch_states = train_states[i:i + batch_size]
        batch_actions = train_actions[i:i + batch_size]
        
        # Forward pass
        predictions = model(batch_states)
        loss = criterion(predictions, batch_actions)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_states)
            val_loss = criterion(val_predictions, val_actions)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Val Loss: {val_loss.item()}')

# Save the model
torch.save(model.state_dict(), '/home/atari_ws/project/BCmodel.pth')
# Save the entire model
#torch.save(model, '/home/atari_ws/project/BCmodel.pth')

# Example usage: Load the model and predict an action
# model = BCModel(input_dim, output_dim)
# model.load_state_dict(torch.load('/home/BCmodel.pth'))
# model = model.to(device)  # Move to GPU if available

# state = torch.rand(1, input_dim).to(device)  # Random state, moved to GPU if available
# model.eval()
# with torch.no_grad():
#     action = model(state)  # Predict the action
#     print(action)

