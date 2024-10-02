import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the dataset
def load_dataset(file_path):
    data = np.load(file_path)
    states = data['states']
    qNext = data['qNext']
    return states, qNext

# Check for missing values
def check_missing_values(states, qNext):
    states_nan = np.isnan(states).sum()
    qNext_nan = np.isnan(qNext).sum()
    print(f"Missing values in states: {states_nan}")
    print(f"Missing values in qNext: {qNext_nan}")
    
# Check for duplicate rows
def check_duplicates(states):
    unique_states = np.unique(states, axis=0)
    duplicates = len(states) - len(unique_states)
    print(f"Duplicate rows in states: {duplicates}")

# Calculate and print basic statistics (mean, std)
def basic_statistics(states, qNext):
    print("States statistics:")
    print(f"Mean: {np.mean(states, axis=0)}")
    print(f"Standard deviation: {np.std(states, axis=0)}")

    print("\nqNext statistics:")
    print(f"Mean: {np.mean(qNext, axis=0)}")
    print(f"Standard deviation: {np.std(qNext, axis=0)}")

# Check correlation between states and qNext
def check_correlation(states, qNext):
    correlation = np.corrcoef(states.T, qNext.T)
    print("\nCorrelation between states and qNext (first few features):")
    print(correlation[:5, :5])  # Print only a subset for clarity

# Check data range and scaling needs
def check_data_range(states, qNext):
    print("\nStates min/max:")
    print(f"Min: {np.min(states, axis=0)}")
    print(f"Max: {np.max(states, axis=0)}")
    
    print("\nqNext min/max:")
    print(f"Min: {np.min(qNext, axis=0)}")
    print(f"Max: {np.max(qNext, axis=0)}")

# Visualize data distribution
def plot_distribution(states, qNext):
    plt.hist(states[:, 0], bins=50, alpha=0.5, label='states feature 0')
    plt.hist(qNext[:, 0], bins=50, alpha=0.5, label='qNext feature 0')
    plt.legend()
    plt.title('Distribution of states and qNext (first feature)')
    plt.show()

# Check feature variance
def check_variance(states):
    variances = np.var(states, axis=0)
    print("\nFeature variances (states):")
    print(variances)

# Split the data into train and validation sets
def split_data(states, qNext):
    states_train, states_val, qNext_train, qNext_val = train_test_split(states, qNext, test_size=0.2, random_state=42)
    print("\nTrain-validation split done. Training set size:", states_train.shape)
    print("Validation set size:", states_val.shape)
    return states_train, states_val, qNext_train, qNext_val

# Main function to run all checks
def main():
    dataset_name = "datasets/55s_noRND.npz"
    
    # Load data
    states, qNext = load_dataset(dataset_name)
    
    # Run checks
    check_missing_values(states, qNext)
    check_duplicates(states)
    basic_statistics(states, qNext)
    check_correlation(states, qNext)
    check_data_range(states, qNext)
    plot_distribution(states, qNext)
    check_variance(states)
    split_data(states, qNext)

if __name__ == "__main__":
    main()
