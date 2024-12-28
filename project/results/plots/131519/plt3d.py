import pickle
import matplotlib.pyplot as plt


# Set up the argument parser
parser = argparse.ArgumentParser(description='name')
parser.add_argument('date', type=str)

# Parse the arguments
args = parser.parse_args()

# Use the provided date from the command-line argument
date = args.date
# Path to the saved .pkl file
file_path = "EE0_trajectory.pkl"
file_path = date
# Load the figure
with open(file_path, 'rb') as f:
    fig = pickle.load(f)

# Ensure Matplotlib uses the correct interactive backend
plt.ion()  # Turn on interactive mode
plt.show(block=True)  # Show the figure interactively

# Optionally, re-display the figure if the backend requires it
fig.canvas.manager.show()

