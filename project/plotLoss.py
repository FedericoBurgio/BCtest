import numpy as np
import matplotlib.pyplot as plt

# Load the loss history data
val = np.load('/home/federico/biconmp_mujoco/project/10-13-25/val_losses.npy')
train = np.load('/home/federico/biconmp_mujoco/project/10-13-25/train_losses.npy')

#loss_history = np.convolve(loss_history, np.ones(100) / 100, mode='valid')
# Plot the loss history on a log scale

plt.plot(val, label='Loss')
plt.plot(train, label='Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.title('Loss History')
plt.legend(['val','tain'])
plt.grid(True)
plt.show()
