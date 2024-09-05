# Importing necessary libraries
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # PyTorch module for neural network layers and loss functions
from torch.utils.data import DataLoader  # DataLoader to handle mini-batching and shuffling of data
from omegaconf import OmegaConf  # For configuration management using OmegaConf

# Importing custom modules likely related to the specific application
from simulation import Simulation  # Custom module for simulation (probably robotic)
from contact_planner import ContactPlanner  # Custom module for contact planning (related to robot movement/contact)
import utils  # Custom utility functions
import pinocchio as pin  # Library for rigid body dynamics and robotics
from database import Database  # Custom module to handle dataset operations

import numpy as np  # Numpy for numerical operations
# Importing matplotlib for plotting (commented out, likely not needed for this part of the script)
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # Backend setting for matplotlib (commented out)
import random  # Random module for reproducibility of results
import hydra  # Hydra for configuration management
import os  # OS module for interacting with the operating system (e.g., file paths)
from tqdm import tqdm  # tqdm for progress bars during loops
from datetime import datetime  # For handling dates and times (e.g., logging)
import h5py  # h5py for working with HDF5 file formats (likely used for datasets)
import pickle  # Pickle for serializing and de-serializing Python objects
import wandb  # Weights and Biases for experiment tracking and logging

# Set random seed for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# Log in to Weights and Biases (WandB) for experiment tracking
wandb.login()


class BehavioralCloning():
    """
    Class for Behavioral Cloning, a method to learn policies by imitating expert actions.
    """

    def __init__(self, cfg):
        # Initialize with a configuration file (cfg)
        self.cfg = cfg

        # Model Parameters from configuration
        self.action_type = cfg.action_type  # Type of actions the model will predict
        self.normalize_policy_input = cfg.normalize_policy_input  # Whether to normalize inputs to the network

        # Data-related parameters
        self.n_state = cfg.n_state  # Number of state features
        self.n_action = cfg.n_action  # Number of actions to predict
        self.goal_horizon = cfg.goal_horizon  # Planning horizon in the context of goals

        # Network-related parameters
        self.criterion = nn.L1Loss()  # Loss function (L1 Loss - mean absolute error)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
        print('Nvidia GPU availability is ' + str(torch.cuda.is_available()))

        # Training properties
        self.n_epoch = cfg.n_epoch  # Number of epochs for training
        self.batch_size = cfg.batch_size  # Batch size for training
        self.n_train_frac = cfg.n_train_frac  # Fraction of data used for training (remaining for validation)
        self.learning_rate = cfg.learning_rate  # Learning rate for the optimizer

    
    def initialize_network(self, input_size=0, output_size=0, num_hidden_layer=3, hidden_dim=512, batch_norm=True):
        """
        Initialize the policy network.

        Args:
            input_size (int): Input dimension size (state + goal). Defaults to 0.
            output_size (int): Output dimension size (action). Defaults to 0.
            num_hidden_layer (int): Number of hidden layers. Defaults to 3.
            hidden_dim (int): Number of nodes per hidden layer. Defaults to 512.
            batch_norm (bool): Whether to apply batch normalization. Defaults to True.

        Returns:
            network: The created PyTorch policy network.
        """
        from networks import GoalConditionedPolicyNet  # Import custom network architecture
        
        # Initialize the network with specified parameters
        network = GoalConditionedPolicyNet(input_size, output_size, num_hidden_layer=num_hidden_layer, 
                                           hidden_dim=hidden_dim, batch_norm=batch_norm).to(self.device)
        print("Policy Network initialized")
        return network  # Return the initialized network
    

    def train_network(self, network, batch_size=256, learning_rate=0.002, n_epoch=150, network_save_frequency=10):
        """
        Train the policy network.

        Args:
            network: The policy network to train.
            batch_size (int): Training batch size. Defaults to 256.
            learning_rate (float): Learning rate for the optimizer. Defaults to 0.002.
            n_epoch (int): Number of epochs for training. Defaults to 150.
            network_save_frequency (int): Frequency of saving the network. Defaults to 10.

        Returns:
            network: The trained policy network.
        """             
        
        # Get the training dataset size (use the whole dataset)
        train_set_size = len(self.database)

        print("Dataset size: " + str(train_set_size))
        print(f'Batch size: {batch_size}')
        print(f'learning rate: {learning_rate}')
        print(f'num of epochs: {n_epoch}')

        # Define training and test set sizes
        n_train = int(self.n_train_frac * train_set_size)  # Calculate number of training samples
        n_test = train_set_size - n_train  # Calculate number of validation samples
        
        print(f'training data size: {n_train}')
        print(f'validation data size: {n_test}')
        
        # Split data into training and validation sets
        train_data, test_data = torch.utils.data.random_split(self.database, [n_train, n_test])
        train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)  # DataLoader for training set
        test_loader = DataLoader(test_data, batch_size, shuffle=True, drop_last=True)  # DataLoader for validation set
        
        # Define the optimizer (Adam) for training
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
            
        tepoch = tqdm(range(n_epoch))  # Progress bar for the number of epochs
        
        # Main training loop
        for epoch in tepoch:
            # Set network to training mode
            network.train()
            
            # Initialize lists to store training and validation losses
            train_loss, valid_loss = [], []
            
            # Train the network
            for x, y in train_loader:  # Iterate over batches in the training set
                self.optimizer.zero_grad()  # Clear gradients from the previous step
                x, y = x.to(self.device).float(), y.to(self.device).float()  # Move data to GPU/CPU
                y_pred = network(x)  # Forward pass: predict output
                loss = self.criterion(y_pred, y)  # Compute loss between predicted and actual output
                
                loss.backward()  # Backward pass: compute gradients
                self.optimizer.step()  # Update network weights
                train_loss.append(loss.item())  # Append training loss for this batch
            
            # Validate the network
            test_running_loss = 0
            network.eval()  # Set network to evaluation mode (disables dropout, etc.)
            for z, w in test_loader:  # Iterate over batches in the validation set
                z, w = z.to(self.device).float(), w.to(self.device).float()  # Move data to GPU/CPU
                w_pred = network(z)  # Forward pass: predict output
                test_loss = self.criterion(w_pred, w)  # Compute loss between predicted and actual output
                valid_loss.append(test_loss.item())  # Append validation loss for this batch
                
            # Calculate average losses for the epoch
            train_loss_avg = np.mean(train_loss)
            valid_loss_avg = np.mean(valid_loss)
            tepoch.set_postfix({'training loss': train_loss_avg, 'validation loss': valid_loss_avg})  # Update progress bar with losses
            
            # Log losses to WandB for tracking
            wandb.log({'Training Loss': train_loss_avg, 'Validation Loss': valid_loss_avg})
            
            # Save network periodically
            if epoch > 0 and epoch % network_save_frequency == 0:
                self.save_network(network, name=self.database.goal_type + '_' + str(epoch))
        
        # Save the final trained network
        self.save_network(network, name=self.database.goal_type + '_' + str(n_epoch))    
            
        return network  # Return the trained network
    
    
    def save_network(self, network, name='policy'):
        """
        Save the trained network to a file.

        Args:
            network: The trained policy network.
            name (str): Name of the file to save the network. Defaults to 'policy'.
        """        
        
        # Construct the save path for the network
        savepath = self.network_savepath + "/" + name + ".pth"
        
        # Create a payload to save
        payload = {'network': network, 'norm_policy_input': None}
        
        # Save normalization parameters if needed
        if self.normalize_policy_input:
            payload['norm_policy_input'] = self.database.get_database_mean_std()
        
        # Save the network and related information to the specified path
        torch.save(payload, savepath)
        print('Network Snapshot saved')
        
        
    def run(self):
        """
        Run the training process.
        """           
        
        # NOTE: Initialize Network
        self.cc_input_size = self.n_state + (self.goal_horizon * 3 * 4)  # Calculate input size for 'cc' network 3 dimwnsioni per 4 limbs
        self.vc_input_size = self.n_state + 5  # Calculate input size for 'vc' network (phi, vx, vy, w)
        
        self.output_size = self.n_action  # Output size is the number of actions to predict
        
        # Initialize the 'vc' policy network
        self.vc_network = self.initialize_network(input_size=self.vc_input_size, output_size=self.output_size, 
                                                  num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
                                                  batch_norm=True)
        
        # Initialize the 'cc' policy network
        self.cc_network = self.initialize_network(input_size=self.cc_input_size, output_size=self.output_size, 
                                                  num_hidden_layer=self.cfg.num_hidden_layer, hidden_dim=self.cfg.hidden_dim,
                                                  batch_norm=True)
        
        # NOTE: Load database
        self.database = Database(limit=self.cfg.database_size, norm_input=self.normalize_policy_input)  # Load database with a limit on its size
        filename = '/home/atari_ws/data/behavior_cloning/trot/bc_single_gait_multi_goal_with_stop/dataset/database_0.hdf5'
        self.database.load_saved_database(filename=filename)  # Load the saved dataset from an HDF5 file
        
        # Setup directory path for saving networks
        directory_path = os.path.dirname(filename)
        self.network_savepath = directory_path + '/../network'
        os.makedirs(self.network_savepath, exist_ok=True)  # Create the directory if it doesn't exist
        
        # NOTE: Train Policy
        wandb.init(project='bc_single_gait_multi_goal_with_stop', config={'goal_type':'cc'}, name='cc_training')  # Initialize WandB for 'cc' training
        print('=== Training CC Policy ===')
        self.database.set_goal_type('cc')  # Set goal type to 'cc'
        self.cc_network = self.train_network(self.cc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch, network_save_frequency=10)
        wandb.finish()  # Finish the WandB session
        
        wandb.init(project='bc_single_gait_multi_goal_with_stop', config={'goal_type':'vc'}, name='vc_training')  # Initialize WandB for 'vc' training
        print('=== Training VC Policy ===')
        self.database.set_goal_type('vc')  # Set goal type to 'vc'
        self.vc_network = self.train_network(self.vc_network, batch_size=self.batch_size, learning_rate=self.learning_rate, n_epoch=self.n_epoch, network_save_frequency=10)
        wandb.finish()  # Finish the WandB session
            
        
@hydra.main(config_path='cfgs', config_name='bc_config')
def main(cfg):
    # Main function to initialize and run the Behavioral Cloning process
    icc = BehavioralCloning(cfg) 
    icc.run()  # Start the training process

if __name__ == '__main__':
    main()  # Execute main function when script is run
