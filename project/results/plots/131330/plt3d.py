import pickle
import matplotlib.pyplot as plt

# Path to the saved .pkl file
file_path = "EE0_trajectory.pkl"

# Load the figure
with open(file_path, 'rb') as f:
    fig = pickle.load(f)

# Ensure Matplotlib uses the correct interactive backend
plt.ion()  # Turn on interactive mode
plt.show(block=True)  # Show the figure interactively

# Optionally, re-display the figure if the backend requires it
fig.canvas.manager.show()

