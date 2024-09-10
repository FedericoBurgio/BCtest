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
#from BC_copy import PolicyNetwork


from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        action_logits = self.fc4(x)
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
        self.state_size = 37
        self.action_size = 12
        
        # Initialize the policy network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(37, 12).to(device)
        
        # Load the trained model
        self.policy.load_state_dict(torch.load("/home/atari_ws/project/best_policy_final.pth"))
        self.policy.eval()  # Set the model to evaluation mode
    
    def get_torques(self,
                   q: np.array,
                   v: np.array,
                   robot_data: Any,
                   **kwargs
                   ):
        # Combine the state information
        controller =kwargs.get('controller', None)  
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        s = np.append(s, controller.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, controller.gait_gen.gait_planner.get_phase(robot_data.time, 0))
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

class TrainedControllerB(BiConMPC):
    def __init__(self,
                 robot: AbstractRobotWrapper,
                 replanning_time=.05,
                 sim_opt_lag=False,
                 **kwargs
                 ) -> None:
        #super().__init__(robot, **kwargs)
        super().__init__(robot, replanning_time=replanning_time, sim_opt_lag=sim_opt_lag, **kwargs)
        self.state_size = 37 #35+2 phase
        self.action_size = 12
        v_des, w_des = np.array([0.3, 0., 0.]), 0
        self.set_command(v_des, w_des)
            # Set gait
        self.set_gait_params(trot)  # Choose between trot, jump and bound
        
        # Initialize the policy network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(37, 12).to(device)
        
        # Load the trained model
        self.policy.load_state_dict(torch.load("/home/atari_ws/project/phase/best_policy_ep100.pth"))
        self.policy.eval()  # Set the model to evaluation mode
    
    def get_torques(self,
                   q: np.array,
                   v: np.array,
                   robot_data: Any,
                   **kwargs
                   ):
        # Combine the state information
        #controller = kwargs.get('controller', None)  
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(s[35])
        print(s[36])
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
        
        #print(torques)
        
        return torques
 
class TrainedControllerPD(BiConMPC):
    def __init__(self,
                 robot: AbstractRobotWrapper,
                 replanning_time=.05,
                 sim_opt_lag=False,
                 #datasetPath = "/home/atari_ws/project/phase/best_policy_ep100.pth",
                 **kwargs
                 ) -> None:
        #super().__init__(robot, **kwargs)
        super().__init__(robot, replanning_time=replanning_time, sim_opt_lag=sim_opt_lag, **kwargs)
        self.state_size = 37 #35+2 phase
        self.action_size = 12
        v_des, w_des = np.array([0.3, 0., 0.]), 0
        self.set_command(v_des, w_des)
            # Set gait
        self.set_gait_params(trot)  # Choose between trot, jump and bound
        
        # Initialize the policy network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(37, 12).to(device)
        
        # Load the trained model
        self.policy.load_state_dict(torch.load("phase/best_policy_ep10.pth"))
        self.policy.eval()  # Set the model to evaluation mode
    
    def get_torques(self,
                   q: np.array,
                   v: np.array,
                   robot_data: Any,
                   **kwargs
                   ):
        # Combine the state information
        #controller = kwargs.get('controller', None)  
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(s[35])
        print(s[36])
        state_tensor = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            # Get action logits from the model
            q_ = self.policy(state_tensor).view(-1).tolist()
        print(q_)
        print(33333)
        kp=2
        kv=.1
        action = np.zeros(len(q_))
        for i in range(len(q_)):
            action[i] = kp * ( q_[i] - q[i]) - kv*v[i]
      
        
        #action# è il tau!
        #1 che cosa devo registrare? q e q_ o q e tau? come faccio l'addestramento: come uso PD dentro l'addestramento se NN è una black box? mi aspetto q_ da NN? q e v hanno dimensioni (originali) diverse (q 19 v 18)come posso sommarle? uso solo posizioni e velocità dei joints?
        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
         'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
        'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        # Map action to torques (example mapping, adjust as needed)
        torques = {}
        for i in range(12):
            torques[keys[i]] = action[i]
        
        print(torques)
        
        return torques

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
            a[i] = q[7+i] + (tau_[i] + kv*v[6+i])/kp
            
        if not np.isnan(a).any():
            self.s_list.append(s)
            self.qNext_list.append(a)
        
    def save_data(self, filename: str = 'DatasetPD.npz') -> None:
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

           
if __name__ == "__main__":
    import random
    mode = 1#0 recording 
    std_deviation = 0.0655# initial state randomness
    if mode==0:  
        i=0
        while i < 19:
        ##### Robot model
            cfg = Go2Config
            robot = MJPinQuadRobotWrapper(
                *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                rotor_inertia=cfg.rotor_inertia,
                gear_ratio=cfg.gear_ratio,
                )

            q = robot.get_state()[0] + np.random.normal(0, std_deviation, size=19)
            v = robot.get_state()[1] + np.random.normal(0, std_deviation, size=18)
            q[0]=0
            q[1]=0
            q[2]=(q[2]+3*0.27)/4
            q[3:6]=1.05*q[3:6]
        
            robot.reset(q,v)
            robot.pin.info()
            robot.mj.info()
            
            ###### Controller
            controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            
            # Set command
            v_des, w_des = np.array([0.3, 0., 0.]), 0
            controller.set_command(v_des, w_des)
            # Set gait
            
            controller.set_gait_params(trot)  # Choose between trot, jump and bound
            #print(controller.gait_gen.gait_planner.get_phase(robot))
            #simulator = Simulator(robot.mj, controller, DataRecorderBC3(phase=controller.gait_gen.gait_planner.get_phase(robot.mj.data.time,0))) 
            simulator = Simulator(robot.mj, controller, DataRecorderPD(controller)) 
            
            sim_time = 2#s
            simulator.run(
                simulation_time=sim_time,
                use_viewer=False,
                real_time=False,
                visual_callback_fn=None,)
            if not simulator.controller.diverged:
                simulator.data_recorder.save_data("DatasetPD.npz")
                i = i + 1
            else:
                print("Diverged")     
    elif mode==1:
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
        controller = TrainedControllerPD(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        simulator = Simulator(robot.mj, controller)
        
            
        #controller = NoController(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        #simulator = Simulator(robot.mj, controller)
        # Visualize contact locations
        # visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        
        sim_time = 2#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
        )
        