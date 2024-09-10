import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Load dataset
data = np.load("datasetPhase.npz")
states = data['states']
actions_t = data['actions']

state_mean = np.mean(states, axis=0)
state_std = np.std(states, axis=0)
#states = (states - state_mean) / (state_std + 1e-6)


# Remove NaN values
valid_mask = ~np.isnan(actions_t).any(axis=1) & ~np.isnan(states).any(axis=1)
states = states[valid_mask]
actions_t = actions_t[valid_mask]

q=np.zeros((len(states),12))
v=np.zeros((len(states),12))
for i in range(len(states)):
    q[i]=states[i][5:17]
    v[i]=states[i][23:35]
print(q.shape)
print(v.shape)
print(actions_t.shape)
print(states.shape)
print(q[7])
print(states[7][5:17])
print(q[7]==states[7][5:17])

kp=2
kv=.1
actions = np.zeros((len(states),12))
for i in range(len(states)):
    for j in range(12):
        actions[i][j] = q[i][j] + (actions_t[i][j] + kv*v[i][j])/kp
#actions = q + (actions_t + kv*v)/kp 
lossH=[]
print(333)
print(actions.shape)
np.save("actions.npy",actions)
ddd
# Convert to PyTorch tensors and move to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
states = torch.tensor(states, dtype=torch.float32).to(device)
actions = torch.tensor(actions, dtype=torch.float32).to(device)

# Create datasets and dataloaders
dataset = TensorDataset(states, actions)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        action_logits = self.fc4(x)
        return action_logits
    
class PolicyNetwork2(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        action_logits = self.fc4(x)
        return action_logits

# Initialize model, loss function, and optimizer
state_size = states.shape[1]
action_size = actions.shape[1]
model = PolicyNetwork(state_size, action_size).to(device)

criterion = nn.MSELoss()  # MSE for continuous actions
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop variables
num_epochs = 1000
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
            predicted_train_actions = model(batch_states)
            train_loss = criterion(predicted_train_actions, batch_actions)
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

        # Store losses
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        # Save best model based on validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_policy = model.state_dict()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_train_loss:.4e}, Val Loss: {average_val_loss:.4e}, Best Val Loss: {best_val_loss:.4e}')
        
        # Save the best policy every 10 epochs
        if epoch % 10 == 0 and epoch != 0:
            model_path = f'phase/best_policy_ep{epoch}.pth'
            torch.save(best_policy, model_path)
            print(f"Model saved to {model_path}")

except KeyboardInterrupt:
    print("Training interrupted. Saving the best model...")

# Save the final best model
model_path = "phase/best_policy_final.pth"
torch.save(best_policy, model_path)
print(f"Final best model saved to {model_path}")
np.save("lossHistory.npy", train_losses)