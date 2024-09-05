import numpy as np
import matplotlib.pyplot as plt

# Load the loss history data
loss_history = np.load('lossHistory.npy')

# Plot the loss history on a log scale

plt.plot(loss_history, label='Loss')
#plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss (log scale)')
plt.title('Loss History')
plt.legend()
plt.grid(True)
plt.show()