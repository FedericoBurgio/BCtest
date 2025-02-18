import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import os
import nets  # Importing your modified PolicyNetwork
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
    #     time_per_recording = 500 #slice the dataset to get only the first time_per_recording timesteps for each recoridng
    #     total_size = f['states'].shape[0]

    #     indices = np.ravel(
    #         np.arange(0, total_size, 1000)[:, None] + np.arange(time_per_recording)
    #     )

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

def preprocess_data(states, qNext, step=1):
    """Downsamples and converts data to PyTorch tensors."""
    states = states[::step]
    qNext = qNext[::step]
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(qNext, dtype=torch.float32).to(device)
    
    # print("NORMALIZZAZIONE")
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # states = scaler.fit_transform(states)
    # states = torch.tensor(states, dtype=torch.float32).to(device)
    
    return states, actions, device

def preprocess_dataNEW(states, qNext, step=1):
    """Downsamples, normalizes, and converts data to PyTorch tensors using pipelines."""
    import numpy as np
    import torch
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    # Downsample the data
    states = states[::step]
    qNext = qNext[::step]
    
    # Define feature indices
    continuous_indices = [i for i in range(0, 35)] + ([36, 37, 38,
    40, 41, 42,
    44, 45, 46,
    48, 49, 50,
    53, 54, 55, 
    57, 58, 59,
    61, 62, 63,
    65, 66, 67,
    78, 79, 80, 81])
    
    boolean_indices = [35, 39, 43, 47, 52, 56, 60, 64, 73, 74, 75, 76]
    time_indices = [51, 68]
    gait_phase_indices = [69, 70, 71, 72]
    gait_index_indices = [77]
    
    
    
    # Remove overlapping indices
    continuous_indices = list(set(continuous_indices) - set(boolean_indices + time_indices + gait_phase_indices + gait_index_indices))
    
    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', StandardScaler(), continuous_indices + time_indices),
            ('boolean', 'passthrough', boolean_indices),
            ('gait_phase', 'passthrough', gait_phase_indices),
            ('gait_index', OneHotEncoder(sparse_output=False), gait_index_indices)
        ])
    
    # Fit and transform the data
    processed_states = preprocessor.fit_transform(states)
    
    import joblib

    # Fit the preprocessor on the training data
    preprocessor.fit(states)

    # Save the preprocessor
    joblib.dump(preprocessor, f"{save_dir}/preprocessor.joblib")
    print(f"Preprocessor saved to {save_dir}/preprocessor.joblib")
        
    # Convert boolean features to integers if necessary
    # Assuming 'passthrough' keeps the data as is, but you may need to convert booleans explicitly
    boolean_start_idx = len(continuous_indices + time_indices)
    boolean_end_idx = boolean_start_idx + len(boolean_indices)
    processed_states[:, boolean_start_idx:boolean_end_idx] = processed_states[:, boolean_start_idx:boolean_end_idx].astype(np.int32)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_tensor = torch.tensor(processed_states, dtype=torch.float32).to(device)
    actions_tensor = torch.tensor(qNext, dtype=torch.float32).to(device)
    
    return states_tensor, actions_tensor, device

def create_dataloaders(states, actions, batch_size=256):
    """Creates train and validation DataLoader objects."""
    states = states.unsqueeze(1)  # Add sequence length dimension
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

# Main script
dataset_name = "1samples_5duration_ForcesTruePertTruedet4_999.h5"  # Specify dataset name
states, qNext = load_data(dataset_name)
save_dir = create_save_folder()

# NON CANCELLARE
# Uncomment and adjust as necessary for dataset preprocessing

# Preprocess data
step = 1

# #NON CANCELLARE
# # #next two lines removes xyz from bool xyz of the actual EE    
#columns_to_remove = np.r_[53:56, 57:60, 61:64, 65:68, 73:78] 
# # states = np.delete(states, columns_to_remove, axis=1)

# # states = np.delete(states,[70,71,72],axis=1)
# #states = np.delete(states,[-5,-6,-7],axis=1)
# #states = np.delete(states,np.r_[70:74],axis=1) #reomves phase 1 2 3 (leave only 0) and gait index
#states = np.delete(states,np.r_[69:78],axis=1) 

# #states = np.delete(states, [-5,-6,-7], axis=1)
states = tuple(
    tuple(state[i] * 0.05 if i == 51 else state[i] for i in range(len(state)))
    for state in states
)

#states = np.delete(states, np.r_[0,1], axis=1)
#[52:69, 73:77, 78:82]
states = np.delete(states, np.r_[52:69, 73:77, 78:82], axis=1)
#states = np.delete(states, np.r_[52:69, 73:78], axis=1)
#states = np.delete(states, np.r_[35:69], axis=1)


# # Remove NaN values
# # valid_mask = ~np.isnan(qNext).any(axis=1) & ~np.isnan(states).any(axis=1)

#states = np.delete(states, columns_to_remove, axis=1)
#breakpoint()
#states = np.delete(states, np.r_[53:56, 57:60, 61:64, 65:68, 73, 74, 75, 76, 77], axis=1) #phase 0,1,2,3, gait index

states, actions, device = preprocess_data(states, qNext, step)

# Create dataloaders
batch_size = 128
train_loader, val_loader, state_size, action_size = create_dataloaders(states, actions, batch_size)

print(f"State size: {state_size}")

# Initialize model, criterion, optimizer, and scheduler
model = nets.PolicyNetworkBUONO(state_size, action_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6)

# Train the model
num_epochs = 250 * step
best_policy, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir)

# Save the best model and losses
save_model_and_losses(best_policy, train_losses, val_losses, save_dir)
