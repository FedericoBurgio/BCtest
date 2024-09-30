import numpy as np
import matplotlib.pyplot as plt

# Load the data from the .npz files
view = (np.load("test_with_view.npz")['states'][4000:5000])[:, :10]
no_view = (np.load("test_no_view.npz")['states'][4000:5000])[:, :10]
#breakpoint()
# # Check if the arrays are the same
# are_equal = np.array_equal(view, no_view)

# # Print whether the arrays are equal
# if are_equal:
#     print("The sliced arrays are the same.")
# else:
#     print("The sliced arrays are not the same.")
print("view shape: ", view.shape)
print("no_view shape: ", no_view.shape)

# Plotting the arrays
plt.figure(figsize=(12, 6))

# Plot view
plt.subplot(1, 2, 1)
plt.plot(view)
plt.title('View States [0:5]')
plt.xlabel('Index')
plt.ylabel('Values')

# Plot no_view
plt.subplot(1, 2, 2)
plt.plot(no_view)
plt.title('No View States [0:5]')
plt.xlabel('Index')
plt.ylabel('Values')

plt.tight_layout()
plt.show()
