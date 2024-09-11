
from typing import Any, Dict, Callable
import numpy as np
import os

class DataRecorderBC3(object):
    def __init__(self, controller, record_dir: str = "/home/atari_ws/project/") -> None:
        self.record_dir = record_dir
        self.controller_=controller
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.a_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        #phase = [self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, i) for i in range(4)]
        
        s = np.concatenate((q[2:], v))  # Create a state from q and v components
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        a = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        if not np.isnan(a).any():
            self.s_list.append(s)
            self.a_list.append(a)
        
    def save_data(self, filename: str = 'DatasetBC.npz') -> None:
        # Convert lists to arrays for efficient storage
        new_states = np.array(self.s_list)
        new_actions = np.array(self.a_list)
        
        # Check if the file already exists
        if os.path.exists(filename):
            # Load existing data
            existing_data = np.load(filename)
            existing_states = existing_data['states']
            existing_actions = existing_data['actions']

            # Append new data to existing data
            combined_states = np.concatenate((existing_states, new_states), axis=0)
            combined_actions = np.concatenate((existing_actions, new_actions), axis=0)
        else:
            # If file does not exist, just use new data
            combined_states = new_states
            combined_actions = new_actions

        # Save the combined data back to the file
        np.savez(filename, states=combined_states, actions=combined_actions)
        print(f"Data saved to {filename}")
 
class DataRecorderPD(object):
    def __init__(self, controller, record_dir: str = "/home/atari_ws/project/") -> None:
        self.record_dir = record_dir
        self.controller_=controller
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.qNext_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        #phase = [self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, i) for i in range(4)]
        
        s = np.concatenate((q[2:], v))  # Create a state from q and v components
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        tau_ = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        kp = 2
        kv = 0.1
        a = np.zeros(12)
        for i in range(12):
            a[i] = q[7+i] + (tau_[i] + kv*v[6+i])/kp# +7: x y z qx qy qz alpha q1 ... q12 q1 index 7; +6 same 
            
        if not np.isnan(a).any():
            self.s_list.append(s)
            self.qNext_list.append(a)
        
    def save_data(self, filename: str = 'DatasetPD2.npz') -> None:
        # Convert lists to arrays for efficient storage
        new_states = np.array(self.s_list)
        new_actions = np.array(self.qNext_list)
        
        # Check if the file already exists
        if os.path.exists(filename):
            # Load existing data
            existing_data = np.load(filename)
            existing_states = existing_data['states']
            existing_actions = existing_data['qNext']

            # Append new data to existing data
            combined_states = np.concatenate((existing_states, new_states), axis=0)
            combined_actions = np.concatenate((existing_actions, new_actions), axis=0)
        else:
            # If file does not exist, just use new data
            combined_states = new_states
            combined_actions = new_actions

        # Save the combined data back to the file
        np.savez(filename, states=combined_states, qNext=combined_actions)
        print(f"Data saved to {filename}")