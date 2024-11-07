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
                 state_size = -1, #35+2+20 cont phase
                 action_size = -1,
                 hL = [],
                 **kwargs
                 ) -> None:
        #super().__init__(robot, **kwargs)
        super().__init__(robot, replanning_time=replanning_time, sim_opt_lag=sim_opt_lag, **kwargs)
        
        self.state_size = state_size
        self.action_size = action_size
        self.hL = hL

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(self.state_size, self.action_size, hidden_layers= [256,256], rnn_hidden_size=256, num_rnn_layers=1).to(device)
        self.gait_index = -1
        
        # Load the trained model
        self.policy.load_state_dict(torch.load(datasetPath))
        self.policy.eval()  # Set the model to evaluation mode
        
        
        self.state_history = deque(maxlen=4)
        
    def get_torques(self,
                q: np.array,
                v: np.array,
                robot_data: Any,
                **kwargs
                ):
        super().get_torques(q, v, robot_data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        
        cnt_base = self.detect_contact_steps3(self.gait_gen.cnt_plan,q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        s = np.append(s, cnt_base[0].flatten()) # actual next contact
        s = np.append(s, cnt_base[1]) # timesteps. Note: NOT expressed in seconds
        if self.state_size == 73 : #bools + xyz degli EE
            tmp = self.gait_gen.cnt_plan[0]
            tmp[:,1:] = self.express_contact_plan_in_consistant_frame(q,tmp[:,1:],True)
            s = np.append(s, tmp.flatten())
            s = np.append(s, 0)
        elif self.state_size == 61: # solo booleani
            s = np.append(s, (self.gait_gen.cnt_plan[0][:,0]).flatten()) #qua va o no il flatten bho
            s = np.append(s, 0)
        elif self.state_size == 75: #
            tmp = self.gait_gen.cnt_plan[0]
            tmp[:,1:] = self.express_contact_plan_in_consistant_frame(q,tmp[:,1:],True)
            s = np.append(s, tmp.flatten())
            s = np.append(s, 0)
            s = np.append(s, self.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
            s = np.append(s, self.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        
        
        s = np.append(s, self.v_des)
        s = np.append(s, self.w_des)

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
    def get_torquesOLD_(self,
                    q: np.array,
                    v: np.array,
                    robot_data: Any,
                    **kwargs
                    ):
        super().get_torques(q, v, robot_data)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s = np.concatenate((q[2:], v))  # Example combination of state variables
        
        # Prepare contact information
        # cnt_wrd = []
        # tmp = self.detect_contact_steps(self.gait_gen.cnt_plan)
        # for i in range(4):  # 4 EE
        #     cnt_wrd.append(tmp[1 + 5 * i:4 + 5 * i])
        
        # cnt_base = self.express_contact_plan_in_consistant_frame(q, np.array(cnt_wrd), base_frame=True)
        # cnt_base_copy = cnt_base.copy()
        
        # for i in range(4):
        #     cnt_base = np.insert(cnt_base, 3 + i * 3 + i, tmp[4 + 5 * i])  # Check indices
        
        # # Concatenate all relevant state information
        cnt_base = self.detect_contact_steps3(self.gait_gen.cnt_plan, q) # #17:[4x4],1 _ [[bool1,x1,y1,z1],[bool2,x2,y2,z2],[bool3,x3,y3,z3],[bool4,x4,y4,z4],timesteps]
        

        s = np.append(s, cnt_base[0].flatten()) # actual next contact
        s = np.append(s, cnt_base[1]) # timesteps. Note: NOT expressed in seconds
        
        s = np.append(s, self.v_des)
        s = np.append(s, self.w_des)

        # Add the current state to the history
        self.state_history.append(s)

        # Check if we have enough timesteps (4)
        if len(self.state_history) < 2:
            # If not enough timesteps, return zero torques or an initial policy
            return {key: 0 for key in ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
                                       'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
                                       'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']}

        # Convert state history to a tensor of shape [1, 4, input_size] for the RNN
        state_tensor = torch.tensor(list(self.state_history), dtype=torch.float32).to(device).unsqueeze(0)

        with torch.no_grad():
            # Get action logits from the model using the sequence of states
            q_ = self.policy(state_tensor).view(-1).tolist()

        # PD control parameters
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
    
    # def get_torques(self,
    #             q: np.array,
    #             v: np.array,
    #             robot_data: Any,
    #             **kwargs
    #             ):
    #     super().get_torques(q, v, robot_data)

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     s = np.concatenate((q[2:], v))  # Example combination of state variables
        
    #     # Prepare contact information
    #     cnt_wrd = []
    #     tmp = self.detect_contact_steps(self.gait_gen.cnt_plan)
    #     for i in range(4):  # 4 EE
    #         cnt_wrd.append(tmp[1 + 5 * i:4 + 5 * i])
        
    #     cnt_base = self.express_contact_plan_in_consistant_frame(q, np.array(cnt_wrd), base_frame=True)
    #     cnt_base_copy = cnt_base.copy()
        
    #     for i in range(4):
    #         cnt_base = np.insert(cnt_base, 3 + i * 3 + i, tmp[4 + 5 * i])  # Check indices
        
    #     # Concatenate all relevant state information
    #     s = np.append(s, cnt_base)
    #     s = np.append(s, self.v_des)
    #     s = np.append(s, self.w_des)

    #     # Convert to tensor and reshape for RNN
    #     state_tensor = torch.tensor(s, dtype=torch.float32).to(device).unsqueeze(0)  # Add batch dimension
    #     state_tensor = state_tensor.unsqueeze(1)  # Shape: [1, 1, input_size] for RNN

    #     with torch.no_grad():
    #         # Get action logits from the model
    #         q_ = self.policy(state_tensor).view(-1).tolist()

    #     kp = 2
    #     kv = 0.1
    #     action = np.zeros(len(q_))
        
    #     for i in range(len(q_)):
    #         action[i] = kp * (q_[i] - q[i + 7]) - kv * v[i + 6]

    #     keys = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',
    #             'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',
    #             'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']
        
    #     torques = {}
    #     for i in range(12):
    #         torques[keys[i]] = action[i]

    #     return torques


    
    def detect_contact_steps3(self, horizon_data, q):
        # Returns the whole predicted state when any boolean changes and the number of time steps it took to get there
        num_time_steps = horizon_data.shape[0]
        num_end_effectors = horizon_data.shape[1]

        initial_booleans = horizon_data[0, :, 0]  # Initial contact states for all end effectors

        for t in range(1, num_time_steps):
            current_booleans = horizon_data[t, :, 0]  # Contact states at time step t

            if not (current_booleans == initial_booleans).all():
                # A change in boolean state detected for at least one end effector

                # Extract the contact plan positions (everything but the boolean column)
                cnt_plan_pos = horizon_data[t, :, 1:]  # Shape [4, 3] (3D positions for each end effector)
                
                # Modify the contact plan using express_contact_plan_in_consistant_frame
                cnt_plan_pos_modified = self.express_contact_plan_in_consistant_frame(q, cnt_plan_pos, base_frame=True)
                
                # Overwrite the original contact plan positions in horizon_data with the modified ones
                horizon_data[t, :, 1:] = cnt_plan_pos_modified  # Only modify the part without the boolean
                
                # Return the modified full state at time step t and the step index
                return horizon_data[t], t

        # If no change in boolean state occurs within the horizon, return the last state and total steps
        cnt_plan_pos = horizon_data[-1, :, 1:]  # Extract last contact plan positions
        cnt_plan_pos_modified = self.express_contact_plan_in_consistant_frame(q, cnt_plan_pos)  # Modify last positions
        horizon_data[-1, :, 1:] = cnt_plan_pos_modified  # Overwrite last contact plan with modified one

        return horizon_data[-1], num_time_steps

    def express_contact_plan_in_consistant_frame(self,q : np.ndarray,
                                            cnt_plan_pos : np.ndarray,
                                            base_frame : bool = False) -> np.ndarray:
        """
        Express contact plan positions in the same frame.
        Gait generator gives contact plan with x, y in base frame and z in world frame.

        Args:
            - q (np.ndarray): state of the robot
            - cnt_plan_pos (np.ndarray): 3D positions of the contact plan.
            x, y are in base frame while z is in world frame.
            shape [H, 4 (feet), 3]
            - base_frame (bool): express contact plan in base frame. Otherwise world frame.

        Returns:
            np.ndarray: cnt_plan_pos in base frame
        """
        # For all points of the cnt plan
        # [x_B, y_B, z_B, 1].T = B_T_W . [x_W, y_W, z_W, 1].T
        # z_B, x_W and y_W are unknown
        # One can express the equality as A.X = B with:
        # X = [x_W, y_W, z_B]
        # A = [[1,0,0], [0,1,0], [0,0,0]] - B_R_W[:, -1]
        # B = W_R_B . ([x_B, y_B, 0].T - W_p_B) - [0, 0, z_W].T
        # (W_p_B = [0., 0., z_B].T as the contact plan is computed with the base at the origin)
        # B = W_R_B . ([x_B, y_B, -z_B].T) - [0, 0, z_W].T
        # Then X = A^{-1}.B
    
        # Reshape to process all positions at once
        cnt_plan_p = cnt_plan_pos.reshape(-1, 3).copy()

        # Rotation matrix W_R_B from world to base
        W_R_B = pinocchio.Quaternion(q[3:7]).matrix()  # Rotation matrix from base to world
        
        # Analytical form of the inverse of A
        A_inv = np.diag([1., 1., 1. / W_R_B[-1, -1]])
        A_inv[-1, :2] = - W_R_B[:2, -1] / W_R_B[-1, -1]

        # Prepare the contact positions for vectorized operation
        p_B = cnt_plan_p.copy()
        p_B[:, -1] = -q[2]  # Set z_B for all points

        # Compute B for all contact points in one operation
        B = W_R_B @ (p_B.T) # Apply rotation to the base frame points
        B = B.T  # Transpose to get shape [N, 3]
        B[:, -1] -= cnt_plan_p[:, -1]  # Subtract z_W from the last coordinate of B

        # Compute X for all positions at once
        X = (A_inv @ B.T).T

        # Apply the final transformations based on the base_frame flag
        if base_frame:
            cnt_plan_p[:, -1] = X[:, -1]  # Update z_B in base frame
        else:
            cnt_plan_p[:, :-1] = X[:, :-1] + q[:2]  # Update x_W and y_W in world frame

        # Reshape back to the original shape [H, 4, 3]
        cnt_plan_p = cnt_plan_p.reshape(-1, 4, 3)

        return cnt_plan_p