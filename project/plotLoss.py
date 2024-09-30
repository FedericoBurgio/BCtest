import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Plot training and validation loss history.')
parser.add_argument('date', type=str, help='The date string for the directory path.')

# Parse the arguments
args = parser.parse_args()

# Use the provided date from the command-line argument
date = args.date

# Load the loss history data
filepath = f'/home/federico/biconmp_mujoco/project/models/{date}'
val = np.load(filepath + '/val_losses.npy')
train = np.load(filepath + '/train_losses.npy')

#loss_history = np.convolve(loss_history, np.ones(100) / 100, mode='valid')
plt.plot(val, label='Loss')
plt.plot(train, label='Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.title('Loss History')
plt.legend(['val','tain'])
plt.grid(True)
plt.show()
