def get_com0():
    import random
    import itertools
    import numpy as np

    #set seed
    random.seed(42)   # Define input ranges
    inside = False   
    vx_in = np.arange(-0.1, 0.6, 0.1)
    vy_in = np.arange(-0.3, 0.4, 0.1)

    # Round the grid values
    vx_in = np.around(vx_in, 1)
    vy_in = np.around(vy_in, 1)

    # Generate combinations for input
    comb_in = list(itertools.product([1, 0], vx_in, vy_in, [0]))
    comb_in = [tuple(c) for c in comb_in]  # Convert to tuples for set operations

    # Define output ranges
    vx_out = np.arange(-0.2, 1.1, 0.1)
    vy_out = np.arange(-0.6, 0.7, 0.1)

    # Round the grid values
    vx_out = np.around(vx_out, 1)
    vy_out = np.around(vy_out, 1)

    # Generate combinations for output
    comb_out = list(itertools.product([1, 0], vx_out, vy_out, [0]))
    comb_out = [tuple(c) for c in comb_out]  # Convert to tuples for set operations

    # Convert lists to sets
    set_in = set(comb_in)
    set_out = set(comb_out)

    # Compute the complementary set
    if inside:
        combinations = set_in
    else:
        combinations = list(set_out - set_in)
    # (Optional) If you need the result as a list of lists
    combinations = [list(c) for c in combinations]
    #combinations = [list(c) for c in comb_in]
    # Create all possible combinations of the elements
    #combinations = list(itertools.product([1, 0], vx, vy, [0]))

    # Convert combinations to lists for easier manipulation
 #   combinations = [list(c) for c in combinations]

    # Perform 20 extractions of comb-like structures
    comb_dict_list = []
    for _ in range(50):
        available_combinations = combinations.copy()
        comb = []
        for _ in range(4):  # Extract 4 unique elements for each comb
            element = random.choice(available_combinations)
            comb.append(element)
            available_combinations.remove(element)
        # Generate a random simulation time between 1.0 and 1.9 (0.1 step)
        sim_time = round(random.uniform(1.0, 1.9), 1)
        comb_dict_list.append({"comb": comb, "sim_time": sim_time})
    j=0
    for i in comb_dict_list:
        print(j,i)
        j+=1            
get_com0()