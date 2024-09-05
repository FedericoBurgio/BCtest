# plot_saved_data.py

import numpy as np
import pickle
import matplotlib.pyplot as plt

def load_and_plot_data(filename: str = 'record.pkl') -> None:
    """Load data from a .pkl file and plot the q and v arrays."""

    # Load the data from the file
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    
    # Extract q and v from the loaded data
    q_list = [entry['q'] for entry in loaded_data]
    v_list = [entry['v'] for entry in loaded_data]

    # Convert lists to numpy arrays for easier plotting
    q_array = np.array(q_list)
    v_array = np.array(v_list)

    # Plot q values
   
    for i in range(q_array.shape[1]):
        plt.plot(q_array[:, i], label=f'q[{i}]')
    #plt.title('Plot of q values over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('q')
    plt.legend()
    plt.show()

    # Plot v values
    #plt.figure(figsize=(10, 5))
    for i in range(v_array.shape[1]):
        plt.plot(v_array[:, i], label=f'v[{i}]')
    #plt.title('Plot of v values over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('v')
    plt.legend()
    plt.show()

    
# Specify the file name where the data is stored
filename = 'record.pkl'

# Call the function to load and plot the data
load_and_plot_data(filename)


