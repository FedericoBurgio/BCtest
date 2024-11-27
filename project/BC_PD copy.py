import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os
import h5py
from time import sleep

def clear_data(states, qNext):
    """Cleans the data by removing invalid (NaN, Inf, or out-of-bound) entries."""
    upper_bound = np.inf
    lower_bound = -np.inf
    valid_indices = ~(
        np.isnan(states).any(axis=1) | np.isinf(states).any(axis=1) |
        np.isnan(qNext).any(axis=1) | np.isinf(qNext).any(axis=1) |
        (np.abs(states) > upper_bound).any(axis=1) |
        (np.abs(qNext) > upper_bound).any(axis=1) |
        (np.abs(states) < lower_bound).any(axis=1) |
        (np.abs(qNext) < lower_bound).any(axis=1)
    )
    return states[valid_indices], qNext[valid_indices]

def load_data(dataset_name):
    """Loads the dataset from an HDF5 file and cleans it."""
    with h5py.File(f"datasets/{dataset_name}", 'r') as f:
        # time_per_recording = 500  # Adjust as needed
        # total_size = f['states'].shape[0]

        # indices = np.ravel(
        #     np.arange(0, total_size, 1000)[:, None] + np.arange(time_per_recording)
        # )

        # Extract the specific indices
        states = f['states'][:]
        qNext = f['qNext'][:]
            
    print(f"Loading dataset: {dataset_name}")
    print(f"Number of samples: {len(states)}")
    sleep(1)
    return clear_data(states, qNext)

def create_save_folder():
    """Creates a folder with a timestamp to save models and data."""
    now = datetime.now()
    date_time_str = now.strftime("%d%H%M")
    save_dir = f"models/{date_time_str}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def create_sequences(states, actions, seq_length):
    """Creates sequences of data for LSTM input."""
    sequences = []
    targets = []
    for i in range(len(states) - seq_length):
        sequences.append(states[i:i+seq_length])
        targets.append(actions[i+seq_length-1])  # Target is the action at the last time step
    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    return sequences, targets

def preprocess_data(states, qNext, step=1, seq_length=10):
    """Downsamples and converts data to PyTorch tensors, creating sequences."""
    states = states[::step]
    qNext = qNext[::step]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(qNext, dtype=torch.float32)

    # Create sequences
    sequences, targets = create_sequences(states, actions, seq_length)

    return sequences.to(device), targets.to(device), device

def create_dataloaders(states, actions, batch_size=256):
    """Creates train and validation DataLoader objects."""
    dataset = TensorDataset(states, actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader, states.shape[2], actions.shape[1]

def save_model_and_losses(best_policy, train_losses, val_losses, save_dir):
    """Saves the best model and loss histories."""
    model_path = f"{save_dir}/best_policy_epfinal.pth"
    torch.save(best_policy, model_path)
    print(f"Final best model saved to {model_path}")

    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
    print(f"Loss history saved to {save_dir}/train_losses.npy and val_losses.npy")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir):
    """Trains the model and saves the best policy based on validation loss."""
    best_val_loss = float('inf')
    best_policy = None
    train_losses = []
    val_losses = []

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

            scheduler.step(average_val_loss)

            # Save losses
            train_losses.append(average_train_loss)
            val_losses.append(average_val_loss)

            # Save best model
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_policy = model

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss:.4e}, "
                  f"Val Loss: {average_val_loss:.4e}, Best Val Loss: {best_val_loss:.4e}")

            # Save model periodically
            if (epoch + 1) % 10 == 0:
                model_path = f'{save_dir}/best_policy_ep{epoch + 1}.pth'
                torch.save(best_policy, model_path)
                print(f"Model saved to {model_path}")

    except KeyboardInterrupt:
        print("Training interrupted. Saving the best model...")

    return best_policy, train_losses, val_losses


# Define the new PolicyNetwork with LSTM and Attention


# Main script
if __name__ == '__main__':
    dataset_name = "1samples_5duration_ForcesTruePertFalsedet3_100_103.h5"  # Specify dataset name
    states, qNext = load_data(dataset_name)
    states = np.delete(states, np.r_[69:78], axis=1)
    save_dir = create_save_folder()

    # Preprocess data
    step = 1
    seq_length = 3  # Adjust the sequence length as needed

    states, actions, device = preprocess_data(states, qNext, step, seq_length)

    # Create dataloaders
    batch_size = 256
    train_loader, val_loader, state_size, action_size = create_dataloaders(states, actions, batch_size)

    print(f"State size (input size): {state_size}")
    print(f"Action size (output size): {action_size}")

    # Initialize model, criterion, optimizer, and scheduler
    hidden_size = 128  # You can adjust this
    num_layers = 1  # You can adjust this
    from nets import PolicyNetworkLSTM
    model = PolicyNetworkLSTM(state_size, action_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)

    # Train the model
    num_epochs = 250 * step
    best_policy, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir)

    # Save the best model and losses
    save_model_and_losses(best_policy, train_losses, val_losses, save_dir)
