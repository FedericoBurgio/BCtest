from typing import Any, Dict, Callable
from mpc_controller.bicon_mpc import BiConMPC
import numpy as np
import torch
import torch.nn as nn
import copy
import pinocchio
from collections import deque

from mj_pin_wrapper.abstract.robot import AbstractRobotWrapper
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract


from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

import nets
from nets import PolicyNetwork
from collections import deque

import detectContact

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
                 state_size = -1, #35+2+20 cont phase
                 action_size = -1,
                 hL = [],
                 **kwargs
                 ) -> None:
        #super().__init__(robot, **kwargs)
        super().__init__(robot, replanning_time=replanning_time, sim_opt_lag=sim_opt_lag, **kwargs)
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.policy = torch.load(datasetPath)        
        self.gait_index = -1
        
        # Load the trained model
        #self.policy.load_state_dict(torch.load(datasetPath))
        self.policy.eval()  # Set the model to evaluation mode
        self.count = 0
    
    def get_torques(self,
                q: np.array,
                v: np.array,
                robot_data: Any,
                **kwargs
                ):
        super().get_torques(q, v, robot_data)
        #self.count = self.count + 1 
        import numpy as np
        import torch
        # import joblib
        
        # q_copy = q.copy()
        # v_copy = v.copy()
        # q_copy[:2] = 0.

        # t = robot_data.time
        # self.robot.update(q_copy, v_copy)
        # cnt_plan = self.gait_gen.compute_raibert_contact_plan(q_copy, v_copy, t, self.v_des, self.w_des)
        # self.gait_gen.cnt_plan = cnt_plan
        # self.robot.update(q, v)
        
        #if self.count % 100 == 0: print(self.gait_gen.cnt_plan) 
        #if self.count == 1000: breakpoint()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        
        cnt_base = detectContact.detect_contact_steps4(self.gait_gen.cnt_plan,q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        s = np.append(s, cnt_base[0].flatten()) # actual next contact
        s = np.append(s, cnt_base[1]) # timesteps. Note: NOT expressed in seconds
        #len 52 
        tmp = self.gait_gen.cnt_plan[0]
        tmp[:,1:] = detectContact.express_contact_plan_in_consistant_frame(q,tmp[:,1:],True)
        s = np.append(s, tmp.flatten())
        s = np.append(s, 0)
        #len 69 nice
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 1))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 2))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 3))
        # #len 73
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 1))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 2))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 3))
        #len 77
        #s = np.append(s, robot_data.time)
        #len 78
        s = np.append(s, self.gait_index)#80,79,78
        s = np.append(s, self.v_des) #81
        s = np.append(s, self.w_des) #82
        #len 83
      #  states = np.delete(states, [73,74,75,76,78], axis=1) #phase 0,1,2,3, gait index
        #s = np.delete(s,np.r_[73:78]) 
        s = np.delete(s, np.r_[73:78])

        # preprocessor = joblib.load("models/201438/preprocessor.joblib")

        # # Transform the state vector
        # processed_states = preprocessor.transform(s.reshape(1, -1))

        # Convert to PyTorch tensor
        #state_tensor = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)

    
    
        # from sklearn.preprocessing import StandardScaler
        # s = StandardScaler().fit_transform(s.reshape(-1,1)).flatten()
        
                
        # Convert to tensor and reshape for RNN
        state_tensor = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
        state_tensor = state_tensor.unsqueeze(1)  # Shape: [1, 1, input_size] for RNN

        with torch.no_grad():
            # Get action logits from the model
            q_ = self.policy(state_tensor).view(-1).tolist()

        kp = 2
        kv = 0.1
        action = np.zeros(len(q_))
        
        for i in range(len(q_)):
            action[i] = kp * (q_[i] - q[i + 7]) - kv * v[i + 6]

        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
                'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
                'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        
        torques = {}
        for i in range(12):
            torques[keys[i]] = action[i]

        return torques
  
class TrainedControllerPDSEQ(BiConMPC):
    def __init__(self,
                 robot: AbstractRobotWrapper,
                 replanning_time=.05,
                 sim_opt_lag=False,
                 datasetPath="",
                 state_size = -1, #35+2+20 cont phase
                 action_size = -1,
                 hL = [],
                 **kwargs
                 ) -> None:
        #super().__init__(robot, **kwargs)
        super().__init__(robot, replanning_time=replanning_time, sim_opt_lag=sim_opt_lag, **kwargs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.seq_length = 3
        self.state_history = deque(maxlen=self.seq_length)
        
        # Load the entire model
        self.policy = torch.load(datasetPath, map_location=device)
          
        self.policy.eval()  # Set the model to evaluation mode

        self.gait_index = -1  # Adjust as needed
    
    
    
    def get_torques(self,
                q: np.array,
                v: np.array,
                robot_data: Any,
                **kwargs
                ):
        super().get_torques(q, v, robot_data)
        import numpy as np
        import torch
        # import joblib
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        
        cnt_base = detectContact.detect_contact_steps3(self.gait_gen.cnt_plan,q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        s = np.append(s, cnt_base[0].flatten()) # actual next contact
        s = np.append(s, cnt_base[1]) # timesteps. Note: NOT expressed in seconds
        #len 52 
        tmp = self.gait_gen.cnt_plan[0]
        tmp[:,1:] = detectContact.express_contact_plan_in_consistant_frame(q,tmp[:,1:],True)
        s = np.append(s, tmp.flatten())
        s = np.append(s, 0)
        #len 69 nice
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 1))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 2))
        s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 3))
        # #len 73
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 1))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 2))
        s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 3))
        #len 77
        #s = np.append(s, robot_data.time)
        #len 78
        s = np.append(s, self.gait_index)#80,79,78
        s = np.append(s, self.v_des) #81
        s = np.append(s, self.w_des) #82
        #len 83
      #  states = np.delete(states, [73,74,75,76,78], axis=1) #phase 0,1,2,3, gait index
        #s = np.delete(s,np.r_[73:78]) 
        states = np.delete(states, np.r_[73:78])
        self.state_history.append(s)

        # Check if we have enough states to form a sequence
        if len(self.state_history) < self.seq_length:
            # Not enough data yet; return zero torques or some default action
            return self.default_torques()
        
        # Create the input sequence tensor
        state_sequence = np.array(self.state_history)  # Shape: (seq_length, state_size)
        state_sequence = torch.tensor(state_sequence, dtype=torch.float32).to(device)
        state_sequence = state_sequence.unsqueeze(0)  # Add batch dimension: (1, seq_length, state_size)

        with torch.no_grad():
            # Get the action from the policy network
            q_pred = self.policy(state_sequence).cpu().numpy().flatten()

        # PD control to compute torques
        kp = 2.0
        kv = 0.1
        action = np.zeros(len(q_pred))
        for i in range(len(q_pred)):
            action[i] = kp * (q_pred[i] - q[i + 7]) - kv * v[i + 6]

        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
                'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
                'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        
        torques = {key: action[i] for i, key in enumerate(keys)}

        return torques

    def default_torques(self):
        # Define default torques when not enough history is available
        keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
                'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
                'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        return {key: 0.0 for key in keys}

