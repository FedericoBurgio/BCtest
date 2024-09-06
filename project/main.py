import copy
import numpy as np
import os
from typing import Any, Dict
#import tensorflow as tf
import torch
import pickle
import time

from mj_pin_wrapper.abstract.robot import AbstractRobotWrapper

from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

import torch
import torch.nn as nn

import torch
from BCtorch import BCModel


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        action_logits = self.fc3(x)
        return action_logits

class NoController(ControllerAbstract):
    def __init__(self,
                 robot: AbstractRobotWrapper,
                 **kwargs
                 ) -> None:
        super().__init__(robot, **kwargs)
    
    def get_torques(self,
                   q: np.array,
                   v: np.array,
                   robot_data: Any,
                   **kwargs
                   ):
        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
         'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
        'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        torques = {}
        for key in keys:
            torques[key] = 0.0   
        return torques

class TrainedController(ControllerAbstract):

    def __init__(self,
                 robot: AbstractRobotWrapper,
                 **kwargs
                 ) -> None:
        super().__init__(robot, **kwargs)
        self.state_size = 35
        self.action_size = 12
        
        # Initialize the policy network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(35, 12).to(device)
        
        # Load the trained model
        self.policy.load_state_dict(torch.load("/home/atari_ws/project/policy_model22.pth"))
        self.policy.eval()  # Set the model to evaluation mode
    
    def get_torques(self,
                   q: np.array,
                   v: np.array,
                   robot_data: Any,
                   **kwargs
                   ):
        # Combine the state information
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_tensor = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Get action logits from the model
            action = self.policy(state_tensor).view(-1).tolist()
        
        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
         'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
        'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        # Map action to torques (example mapping, adjust as needed)
        torques = {}
        for i in range(12):
            torques[keys[i]] = action[i]
        
        print(torques)
        
        return torques
    
class DataRecorderBC(object):
    def __init__(self,
                 record_dir:str="/home/federico") -> None:
        self.record_dir = record_dir
        self.data_list = []
    
        #self.q_list = [] 
        #self.v_list = []
        self.s_list = []
        self.a_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        data_to_save = {
            'q': q,
            'v': v,
            'robot_data': robot_data,
            **kwargs
        }
        s=np.concatenate((q[2:],v[:6]))#z + quaternion #vel + vel angolari
        print(len(s))
        a=np.array(list(tau.values()))
        
        #self.q_list.append(q)
        #self.v_list.append(v)
        #self.data_list.append(data_to_save)
        
        self.s_list.append(s) 
        self.a_list.append(a)
        
        data_to_save_BC = 0
        self.data_list.append({'s':s,'a':a,})

    def save_data(self, filename: str = 'DatasetBC.pkl') -> None:
        # Save the accumulated data to a file
        with open("/home/atari_ws/project/DatasetBC.pkl", 'wb') as file:
            pickle.dump(self.data_list, file)
        print(f"Data saved to {filename}")

class DataRecorderBC2(object):
    def __init__(self, record_dir: str = "/home/atari_ws/project/") -> None:
        self.record_dir = record_dir
        
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.a_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        # Combine state (q and v) and action (tau)
        s = np.concatenate((q[2:], v[:6]))  # Create a state from q and v components
        print(tau)
        a = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        
        # Optionally normalize state/action if needed (example shown below)
        # s = (s - s.mean()) / s.std()  # Example normalization (optional)
        
        # Add the state and action to the list
        self.s_list.append(s)
        self.a_list.append(a)
        # print(a)
        # keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
        #  'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
        # 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        # # Map action to torques (example mapping, adjust as needed)
        # torques = {}
        # for i in range(12):
        #     torques[keys[i]] = a[i]
        
        # print(torques)
        # kkkk
        # Print shape for debugging
        print(f"State shape: {s.shape}, Action shape: {a.shape}")

    def save_data(self, filename: str = 'DatasetBC.npz') -> None:
        # Convert lists to arrays for efficient storage
        states = np.array(self.s_list)
        actions = np.array(self.a_list)
        
        # Save data as a compressed NumPy file
        np.savez_compressed(os.path.join(self.record_dir, filename), states=states, actions=actions)
        
        print(f"Data saved to {filename}")

class DataRecorderBC3(object):
    def __init__(self, record_dir: str = "/home/atari_ws/project/") -> None:
        self.record_dir = record_dir
        
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.a_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        # Combine state (q and v) and action (tau)
        
        s = np.concatenate((q[2:], v))  # Create a state from q and v components
        a = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        if not np.isnan(a).any():
            self.s_list.append(s)
            self.a_list.append(a)
        #print(f"State shape: {s.shape}, Action shape: {a.shape}")
        

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

    def save_dataOLD(self, filename: str = 'DatasetBC_old.npz') -> None:
        # Convert lists to arrays for efficient storage
        states = np.array(self.s_list)
        actions = np.array(self.a_list)
        
        np.savez_compressed(os.path.join(self.record_dir, filename), states=states, actions=actions)
        
        print(f"Data saved to {filename}")
    
    # def save_data(self, filename: str = 'DatasetBC.npz') -> None:
    #     # Convert lists to arrays for efficient storage
    #     states = np.array(self.s_list)
    #     actions = np.array(self.a_list)
        
    #     # Save data as a compressed NumPy file
    #     np.savez_compressed(os.path.join(self.record_dir, filename), states=states, actions=actions)
        
    #     print(f"Data saved to {filename}")
              
if __name__ == "__main__":
    import random
    if False:
        ###### Robot model
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
           
        useMPC=False
        if useMPC: 
            ###### Controller
            controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            
            # Set command
            v_des, w_des = np.array([0.3, 0., 0.]), 0
            controller.set_command(v_des, w_des)
            # Set gait
            controller.set_gait_params(trot)  # Choose between trot, jump and bound
            simulator = Simulator(robot.mj, controller, DataRecorderBC3())
        else:
            controller = TrainedController(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            simulator = Simulator(robot.mj, controller)
        
    
    #controller = NoController(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    #simulator = Simulator(robot.mj, controller)
    # Visualize contact locations
    # visual_callback = (lambda viewer, step, q, v, data :
    #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
    # Run simulation
    
        sim_time = 1.2#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
        )
        if not simulator.controller.diverged:
            simulator.data_recorder.save_data()    
    
    
    
    for i in range(15):
        print(i)
        ###### Robot model
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
        #print(robot.get_state())
        print(robot.get_state()[0])
        print(robot.get_state()[1])
        std_deviation = 0.05 # You can adjust this value to control the amount of randomness

        print(444)
        # Add Gaussian noise to the vector
        q = robot.get_state()[0] + np.random.normal(0, std_deviation, size=19)
        v = robot.get_state()[1] + np.random.normal(0, std_deviation, size=18)
        q[0]=0
        q[1]=0
        q[2]=0.27
        robot.reset(q,v)
        robot.pin.info()
        robot.mj.info()
        
        useMPC=0
        if useMPC: 
            ###### Controller
            controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            
            # Set command
            v_des, w_des = np.array([0.3, 0., 0.]), 0
            controller.set_command(v_des, w_des)
            # Set gait
            controller.set_gait_params(trot)  # Choose between trot, jump and bound
            simulator = Simulator(robot.mj, controller, DataRecorderBC3())
        else:
            controller = TrainedController(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            simulator = Simulator(robot.mj, controller)
        
    #controller = NoController(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    #simulator = Simulator(robot.mj, controller)
    # Visualize contact locations
    # visual_callback = (lambda viewer, step, q, v, data :
    #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
    # Run simulation
    
        sim_time = 1.1#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
        )
        
    
        if useMPC and not simulator.controller.diverged:
            simulator.data_recorder.save_data("dataset20.npz")
        else:
            print("Diverged")
            i = i - 1
        