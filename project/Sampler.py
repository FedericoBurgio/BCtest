import numpy as np
import itertools
from scipy.stats import qmc

def LHS(gaitsI, vspacex, vspacey,wspace,n_samples, seed_ = 10):
    # Define variable ranges

    # gaitsI = np.array([0])
    # vspacex = np.arange(-0.1, 0.6, 0.1)
    # vspacey = np.arange(-0.4, 0.5, 0.1)
    # wspace = np.arange(-0.07, 0.14, 0.07)

    # Create combinations (not used in LHS but kept for reference)
    #comb = list(itertools.product([0, 1], vspacex, vspacey, wspace))

    # Variable arrays
    x1_values = gaitsI  # Example: values from 0 to 4
    x2_values = vspacex
    x3_values = vspacey
    x4_values = wspace

    # Number of samples
    #n_samples = 100  # Adjust as needed

    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=4, seed = seed_)
    sample = sampler.random(n_samples)

    # Map samples to variable ranges

    # Mapping for x1
    x1_indices = np.floor(sample[:, 0] * len(x1_values)).astype(int)
    x1_indices = np.clip(x1_indices, 0, len(x1_values) - 1)
    x1_samples = x1_values[x1_indices]

    # Mapping for x2
    x2_indices = np.floor(sample[:, 1] * len(x2_values)).astype(int)
    x2_indices = np.clip(x2_indices, 0, len(x2_values) - 1)
    x2_samples = x2_values[x2_indices]

    # Mapping for x3
    x3_indices = np.floor(sample[:, 2] * len(x3_values)).astype(int)
    x3_indices = np.clip(x3_indices, 0, len(x3_values) - 1)
    x3_samples = x3_values[x3_indices]

    # Mapping for x4
    x4_indices = np.floor(sample[:, 3] * len(x4_values)).astype(int)
    x4_indices = np.clip(x4_indices, 0, len(x4_values) - 1)
    x4_samples = x4_values[x4_indices]
    
    # Combine samples
    for i in range(len(x1_samples)):
        x1_samples[i] = int(x1_samples[i])
        
    lhs_samples = np.column_stack((x1_samples, x2_samples, x3_samples, x4_samples))
    
    verbose = False
    if verbose:
        print(lhs_samples)
        import matplotlib.pyplot as plt

        variables = ['x1', 'x2', 'x3', 'x4']
        samples = [x1_samples, x2_samples, x3_samples, x4_samples]

        for i, var in enumerate(variables):
            plt.figure(figsize=(6, 4))
            plt.hist(samples[i], edgecolor='k', alpha=0.7)
            plt.title(f'Histogram of {var}')
            plt.xlabel(f'{var} values')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            # Output samples

    return lhs_samples

def sobolOLD(gaitsI, vspacex, vspacey, wspace, n_samples, seed_=10):
    # Variable arrays
    x1_values = gaitsI
    x2_values = vspacex
    x3_values = vspacey
    x4_values = wspace

    # Initialize Sobol sampler
    sampler = qmc.Sobol(d=4, scramble=True, seed=seed_)
    
    # Generate samples in [0,1]^4
    sample = sampler.random(n_samples)

    # Map samples to variable ranges

    # Mapping for x1
    x1_indices = np.floor(sample[:, 0] * len(x1_values)).astype(int)
    x1_indices = np.clip(x1_indices, 0, len(x1_values) - 1)
    x1_samples = x1_values[x1_indices]

    # Mapping for x2
    x2_indices = np.floor(sample[:, 1] * len(x2_values)).astype(int)
    x2_indices = np.clip(x2_indices, 0, len(x2_values) - 1)
    x2_samples = x2_values[x2_indices]

    # Mapping for x3
    x3_indices = np.floor(sample[:, 2] * len(x3_values)).astype(int)
    x3_indices = np.clip(x3_indices, 0, len(x3_values) - 1)
    x3_samples = x3_values[x3_indices]

    # Mapping for x4
    x4_indices = np.floor(sample[:, 3] * len(x4_values)).astype(int)
    x4_indices = np.clip(x4_indices, 0, len(x4_values) - 1)
    x4_samples = x4_values[x4_indices]
    
    # Ensure x1_samples are integers
    x1_samples = x1_samples.astype(int)
        
    sobol_samples = np.column_stack((x1_samples, x2_samples, x3_samples, x4_samples))
    
    verbose = False
    if verbose:
        print(sobol_samples)
        import matplotlib.pyplot as plt

        variables = ['x1', 'x2', 'x3', 'x4']
        samples = [x1_samples, x2_samples, x3_samples, x4_samples]

        for i, var in enumerate(variables):
            plt.figure(figsize=(6, 4))
            plt.hist(samples[i], edgecolor='k', alpha=0.7)
            plt.title(f'Histogram of {var}')
            plt.xlabel(f'{var} values')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

    return sobol_samples

def sobol(n_samples, l_bounds, u_bounds, seed_): 
    # Define the number of dimensions and samples
    n_dimensions = len(l_bounds)
    # n_samples = 30

    # # Define the bounds for each dimension
    # l_bounds = [0, -0.1, -0.3, -0.07]  # Lower bounds for each dimension
    # u_bounds = [1, 0.5, 0.3, 0.07] # Upper bounds for each dimension

    #l_bounds = [0, -0.1, -0.01, 0.01]  # Lower bounds for each dimension
    #u_bounds = [1, 0.6, 0.01, 0.02] # Upper bounds for each dimension

    sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed_)
    samples = sampler.random(n=n_samples)

    # Scale samples
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

    # Round gait index to discrete values
    scaled_samples[:, 0] = (np.round(scaled_samples[:, 0])).astype('int')

    verbose = 0
    if verbose:
        print(scaled_samples)
        import matplotlib.pyplot as plt

        variables = ['x1', 'x2', 'x3', 'x4']
        samples = [scaled_samples[:, 0], scaled_samples[:, 1], 
                   scaled_samples[:, 2], scaled_samples[:, 3]]

        for i, var in enumerate(variables):
            plt.figure(figsize=(6, 4))
            plt.hist(samples[i], edgecolor='k', alpha=0.7)
            plt.title(f'Histogram of {var}')
            plt.xlabel(f'{var} values')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            # Output samples

    return scaled_samples

# gaitsI = np.array([0])
# vspacex = np.arange(-0.1, 0.6, 0.1)
# vspacey = np.arange(-0.4, 0.5, 0.1)
# wspace = np.arange(-0.07, 0.14, 0.07)
# mmm=SampledVelSpace(gaitsI, vspacex, vspacey, wspace,20)
