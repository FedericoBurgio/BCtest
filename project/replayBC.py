import torch
import torch.nn as nn
import numpy as np
import os
from datetime import datetime
from main import SimManager
import nets  # Import your network class from the nets module

# Replay function for training with different step sizes
def replay_with_step(step):
    # Get the current date and time
    now = datetime.now()
    date_time_str = now.strftime("%d%H%M")  # Format as DAY-HOUR-MINUTE

    # Create the directory for saving models if it doesn't exist
    save_dir = "models_replay/" + f"{date_time_str}_step{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    data = np.load("v_w_gaits-jump-trot.npz")
    states = data['states']
    qNext = data['qNext']

    # Remove NaN values
    valid_mask = ~np.isnan(qNext).any(axis=1) & ~np.isnan(states).any(axis=1)
    states = states[valid_mask]
    qNext = qNext[valid_mask]

    # Apply the dynamic step size
    states = states[::step]
    qNext = qNext[::step]

    # Convert to PyTorch tensors and move to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(qNext, dtype=torch.float32).to(device)

    # Create datasets and dataloaders
    dataset = torch.utils.data.TensorDataset(states, actions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    # Define the policy network
    state_size = states.shape[1]
    action_size = actions.shape[1]
    model = nets.PolicyNetwork(state_size, action_size).to(device)

    criterion = nn.MSELoss()  # Loss function: Mean Squared Error
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Adjust the number of epochs based on the step size
    num_epochs = 250 * step
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
                
                # Add Gaussian noise during training
                batch_states = batch_states + torch.normal(mean=0, std=0.01, size=batch_states.shape).to(device)

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
            if (epoch + 1) % 10 == 0:
                model_path = f'{save_dir}/best_policy_ep{epoch+1}.pth'
                torch.save(best_policy, model_path)
                print(f'Model saved to {model_path}')

    except KeyboardInterrupt:
        print("Training interrupted. Saving the best model...")

    # Save the final best model with the dynamic file path
    model_path = f"{save_dir}/best_policy_final.pth"
    torch.save(best_policy, model_path)
    print(f"Final best model saved to {model_path}")

    # Save loss history as numpy arrays
    np.save(f"{save_dir}/train_losses.npy", np.array(train_losses))
    np.save(f"{save_dir}/val_losses.npy", np.array(val_losses))
    print(f"Loss history saved to {save_dir}/train_losses.npy and val_losses.npy")

# Running the replay with step changes (2 to 5)
if __name__ == "__main__":
    for step in range(2, 6):  # Step size from 2 to 5
        print(f"\n\nReplaying with step {step}\n")
        replay_with_step(step)
