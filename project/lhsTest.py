import numpy as np
from scipy.stats import qmc

# Define the ranges for vspacex, vspacey, and wspace
vspacex = np.arange(0, 11, 1)  # 8
vspacey = np.arange(0, 11, 1)  # 9

# Create the bounds of your 3D space
# The bounds will be the min and max of each space
bounds = np.array([
    [vspacex.min(), vspacex.max()],  # bounds for vspacex
    [vspacey.min(), vspacey.max()],  # bounds for vspacey
         # bounds for wspace
])

# Number of samples to generate
n_samples = 10  # You can change this to any number of desired samples

# Generate the LHS samples
sampler = qmc.LatinHypercube(d=len(bounds))  # d is the number of dimensions
lhs_sample = sampler.random(n=n_samples)

# Scale the samples to fit the bounds
scaled_sample = qmc.scale(lhs_sample, bounds[:, 0], bounds[:, 1])

# Print the generated LHS samples
print(vspacex)
print(vspacey)
print(scaled_sample)
breakpoint()
