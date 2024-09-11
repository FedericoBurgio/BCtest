from typing import Any, Dict, Callable
from mpc_controller.bicon_mpc import BiConMPC
import numpy as np
import torch
import torch.nn as nn

from mj_pin_wrapper.abstract.robot import AbstractRobotWrapper
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract


from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from nets import PolicyNetwork

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

class TrainedControllerPD(BiConMPC):
    def __init__(self,
                 robot: AbstractRobotWrapper,
                 replanning_time=.05,
                 sim_opt_lag=False,
                 datasetPath="",
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
        self.policy.load_state_dict(torch.load(datasetPath))
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
    
        kp=2
        kv=.1
        action = np.zeros(len(q_))
        for i in range(len(q_)):
            action[i] = kp * ( q_[i] - q[i+7]) - kv*v[i+6]
      
        
        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
         'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
        'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
         
        torques = {}
        for i in range(12):
            torques[keys[i]] = action[i]
        
        print(torques)
        
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
 