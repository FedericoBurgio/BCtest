# config.py
policy = "081059" #081059: ibrido. 131519: vel. 131330: contact
ep = "final"
#1 2
#comb = [[1, 0.5, 0.1, 0.0],[1, -0.1, 0.1, 0],[0, 0.3, 0, 0],[1, 0.3, -0.3, 0]] #seq1 2.2
# comb = [[1, 0.5, 0.1, 0],[1, 0.3, 0.3, 0],[0, 0, 0, 0],[1, 0.3, -0.3, 0]] #seq2 1.2
# sim_time = 1.2
# # #3 4
# comb = [[0, 0, 0.3, 0],[1, 0.3, 0.3, 0],[0, 0.5, 0, 0],[1, 0.1, -0.3, 0]] #seq3 1.2
# sim_time = 1.2
# comb = [[0, 0, 0.3, 0],[1, 0.3, 0.3, 0],[0, 0.5, 0, 0],[1, 0.1, -0.3, 0]] #seq4 1.6
# sim_time = 1.6

# # 5 6 
# comb = [[0, -0.1, 0.3, 0],[0, 0, 0, 0],[0, 0.5, 0, 0],[1, 0.1, -0.3, 0]] #seq5 2
# sim_time = 2
# comb = [[1, 0.2, 0.2, 0],[1, 0.3, 0.3, 0],[0, 0.5, 0, 0],[0, 0.1, -0.3, 0]] #seq6 1.75
# sim_time = 1.75

# comb = [[1, 1, 0, 0],[1, 0, 0.6, 0],[1, 0, -0.6, 0],[1, -0.2, 0, 0],[1, 0.8, 0.5, 0]] 
# comb = [[1,0.8,0.5,0]]
# comb = [[1, -0.1, 0.3, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]]
# sim_time = 0.45
# #,[1, 0, -0.2, 0]] 
# comb = [[0, 0.3, 0, 0]]
 
##PER HEATMAPS 

import itertools
import numpy as np
vx = np.arange(-0.2,1.1,0.1) #vx out
vy = np.arange(-0.6,0.7,0.1) #vy out
# 
# vx = np.arange(-0.1, 0.6, 0.1) #vx in
# vy = np.arange(-0.3, 0.4, 0.1) #vy in
# 
vx = np.around(vx, 1)
vy = np.around(vy, 1)
comb = list(itertools.product([1,0],vx,vy,[0]))
comb = [list(c) for c in comb]
sim_time = 10

# # PER AGGRE
# def get_com0():
#     import random
#     import itertools
#     import numpy as np

#     #set seed
#     random.seed(42)   # Define input ranges
#     inside = False   
#     vx_in = np.arange(-0.1, 0.6, 0.1)
#     vy_in = np.arange(-0.3, 0.4, 0.1)

#     # Round the grid values
#     vx_in = np.around(vx_in, 1)
#     vy_in = np.around(vy_in, 1)

#     # Generate combinations for input
#     comb_in = list(itertools.product([1, 0], vx_in, vy_in, [0]))
#     comb_in = [tuple(c) for c in comb_in]  # Convert to tuples for set operations

#     # Define output ranges
#     vx_out = np.arange(-0.2, 1.1, 0.1)
#     vy_out = np.arange(-0.6, 0.7, 0.1)

#     # Round the grid values
#     vx_out = np.around(vx_out, 1)
#     vy_out = np.around(vy_out, 1)

#     # Generate combinations for output
#     comb_out = list(itertools.product([1, 0], vx_out, vy_out, [0]))
#     comb_out = [tuple(c) for c in comb_out]  # Convert to tuples for set operations

#     # Convert lists to sets
#     set_in = set(comb_in)
#     set_out = set(comb_out)

#     # Compute the complementary set
#     if inside:
#         combinations = set_in
#     else:
#         combinations = list(set_out - set_in)
#     # (Optional) If you need the result as a list of lists
#     combinations = [list(c) for c in combinations]
#     #combinations = [list(c) for c in comb_in]
#     # Create all possible combinations of the elements
#     #combinations = list(itertools.product([1, 0], vx, vy, [0]))

#     # Convert combinations to lists for easier manipulation
#  #   combinations = [list(c) for c in combinations]

#     # Perform 20 extractions of comb-like structures
#     comb_dict_list = []
#     for _ in range(50):
#         available_combinations = combinations.copy()
#         comb = []
#         for _ in range(4):  # Extract 4 unique elements for each comb
#             element = random.choice(available_combinations)
#             comb.append(element)
#             available_combinations.remove(element)
#         # Generate a random simulation time between 1.0 and 1.9 (0.1 step)
#         sim_time = round(random.uniform(1.0, 1.9), 1)
#         comb_dict_list.append({"comb": comb, "sim_time": sim_time})

#     try:
#         data = np.load("temp/combindex_switch.npz")
#         combindex = data['combindex']
#         combindex += 1
#     except FileNotFoundError:
#         combindex = 0  
#     np.savez("temp/combindex_switch.npz", combindex=combindex)
#     return comb_dict_list[combindex]['comb'], comb_dict_list[combindex]['sim_time']

# comb, sim_time = get_com0()

# comb= [[0, 0.1, 0.3, 0], [1, 0.0, 0.1, 0], [0, 0.3, -0.3, 0], [0, -0.1, 0.3, 0]] #3 index:2 :)) in
# sim_time = 1.0 + 5
# comb = [[1, -0.1, 0.2, 0], [0, 0.4, -0.2, 0], [1, 0.3, -0.1, 0], [1, 0.4, 0.1, 0]] #13 
# sim_time = 1.9 + 5
# comb = [[1, 0.3, 0.3, 0], [0, 0.4, 0.3, 0], [0, 0.4, 0.1, 0], [0, 0.3, 0.3, 0]] #16
# sim_time = 1.1 + 5
# comb = [[1, 0.1, 0.0, 0], [0, 0.1, 0.0, 0], [0, 0.1, -0.2, 0], [1, 0.0, 0.1, 0]] #25
# sim_time =  1.7 + 5

#37
# comb = [[1, 0.4, 0.1, 0], [1, 0.3, -0.1, 0], [1, 0.0, -0.3, 0], [1, 0.3, 0.1, 0]]
# sim_time = 1.8 + 5

#QUA SONO GLI INDEX - out
# comb = [[0, 0.4, -0.4, 0], [1, -0.2, -0.6, 0], [1, 0.9, 0.2, 0], [1, 0.9, -0.1, 0]] #2 
# sim_time = 1.4 + 5

# comb = [[1, 0.3, -0.5, 0], [1, 1.0, -0.5, 0], [1, 0.4, -0.4, 0], [1, 0.6, 0.1, 0]] #16
# sim_time = 5 + 1.6

# comb= [[0, 0.8, 0.3, 0], [0, 1.0, 0.3, 0], [1, -0.2, -0.1, 0], [1, 0.3, 0.5, 0]] #19
# sim_time= 1.2 + 5 

# comb = [[1, -0.2, 0.6, 0], [1, 0.9, -0.2, 0], [1, 0.9, -0.5, 0], [1, 0.7, 0.1, 0]]#43
# sim_time = 1.1 + 5


multi = False #se True valuta le commanded una ad una - SETTARE TRUE PER FARE RECORDING PER HEATMAPS
view = True