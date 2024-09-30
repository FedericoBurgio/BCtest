from typing import Any, Dict, Callable
import numpy as np
import os
import pinocchio

class DataRecorderPD(object):
    def __init__(self, controller, record_dir = "") -> None:
        self.record_dir = record_dir
        self.controller_ = controller
        self.gait_index = -1
        
        # Initialize as empty arrays instead of lists for more efficient storage
        self.s_list = []
        self.qNext_list = []
        print("RECORDER INITIATED")
        
    def record(self, q: np.array, v: np.array, tau: np.array, robot_data: Any, **kwargs) -> None:
        #phase = [self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, i) for i in range(4)]
        
        s = np.concatenate((q[2:], v))  # Create a state from q and v components
        #s = np.append(s, self.controller_.gait_gen.gait_planner.get_percent_in_phase(robot_data.time, 0))
        #s = np.append(s, self.controller_.gait_gen.gait_planner.get_phase(robot_data.time, 0))
        
        #tmp = np.array(self.detect_contact_steps(self.controller_.gait_gen.cnt_plan))
        cnt_wrd = []
        tmp = self.detect_contact_steps(self.controller_.gait_gen.cnt_plan)
        for i in range(4): #4 EE
            #tmp[1+5*i:4+5*i] = self.express_contact_plan_in_consistant_frame(q, tmp[1+5*i:4+5*i], base_frame=True)
            cnt_wrd.append((tmp)[1+ 5*i:4 +5*i])
            #cnt_frame.append(tmp[4 + 5*i])
        cnt_base = self.express_contact_plan_in_consistant_frame(q, np.array(cnt_wrd), base_frame=True)
        #breakpoint()
        cnt_base_copy = cnt_base.copy()
        for i in range(4):
            cnt_base = np.insert(cnt_base, 3 + i*3+i, tmp[4 + 5*i]) # bho double check sti indici #nota ho fatto double ceck sembra tutto ok
    
        s = np.append(s, cnt_base)
        s = np.append(s, self.controller_.v_des)
        s = np.append(s, self.controller_.w_des)
        s = np.append(s, self.gait_index)

        #self.express_contact_plan_in_consistant_frame(q, self.controller_.gait_gen.cnt_plan, base_frame=False)
        #print(tmp)
        
        #breakpoint()
        #s = np.append(s, self.detect_contact_steps(self.controller_.gait_gen.cnt_plan))
        #s = np.append(s, tmp)
        #breakpoint()
        tau_ = np.array(list(tau.values()))    # Convert action (tau) to numpy array
        kp = 2
        kv = 0.1
        
        a = np.zeros(12)
        for i in range(12):
            a[i] = q[7+i] + (tau_[i] + kv*v[6+i])/kp# +7: x y z qx qy qz alpha q1 ... q12 q1 index 7; +6 same 
   
        if not np.isnan(a).any():
            self.s_list.append(s)
            self.qNext_list.append(a)
        
    def save_data(self, filename) -> None:
        # Convert lists to arrays for efficient storage
        new_states = np.array(self.s_list)
        new_actions = np.array(self.qNext_list)
        
        # Check if the file already exists
        if os.path.exists(filename):
            with np.load(filename) as existing_data:
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

    def detect_contact_sequences(self, horizon_data): # in the sequence t0:1 t1:1 0 0 0 t5:1 find the next 1 and store the state and the number of steps taken to get there
        output_list = []                              #i.e. what i understood from majid's explanation

        num_time_steps = horizon_data.shape[0]
        num_end_effectors = horizon_data.shape[1]

        for ee in range(num_end_effectors):
            current_contact = horizon_data[0, ee, 0]  # Start from the first boolean value
            in_contact = current_contact == 1  # Check if we're in contact or not
            step_counter = 0  # Track the number of steps to the next contact

            for t in range(1, num_time_steps):
                step_counter += 1
                current_boolean = horizon_data[t, ee, 0]

                if in_contact:
                    # We are in contact (current boolean is 1)
                    if current_boolean == 0:
                        # Break detected (1 -> 0), now wait for the next 1
                        in_contact = False
                    continue

                # We are in no contact (0), detect the next 1
                if current_boolean == 1:
                    # Store the boolean, x, y, z, and steps taken to get here
                    output_list.extend([current_boolean, horizon_data[t, ee, 1], horizon_data[t, ee, 2], horizon_data[t, ee, 3], step_counter])
                    in_contact = True  # Now we're in contact
                    step_counter = 0  # Reset step counter after detecting a new 1

            # If no 1 was found after the last 0, append the state and total steps
            if not in_contact:
                last_state = horizon_data[-1, ee]
                output_list.extend([last_state[0], last_state[1], last_state[2], last_state[3], num_time_steps])

        return output_list

    def detect_contact_steps(self, horizon_data):#returns current pos if in contact, next pos if not in contact
                                                 #ie what i understood from victor's explanation
        output_list = []

        num_time_steps = horizon_data.shape[0]
        num_end_effectors = horizon_data.shape[1]

        for ee in range(num_end_effectors):
            for t in range(num_time_steps):
                current_boolean = horizon_data[t, ee, 0]
                
                if current_boolean == 1:
                    # If in contact, return the current time step and stop checking for this EE
                    output_list.extend([current_boolean, horizon_data[t, ee, 1], horizon_data[t, ee, 2], horizon_data[t, ee, 3], t])
                    break
                else:
                    # If not in contact, look for the next contact (next 1)
                    for future_t in range(t + 1, num_time_steps):
                        if horizon_data[future_t, ee, 0] == 1:
                            # Found the next contact, return this future time step
                            output_list.extend([1, horizon_data[future_t, ee, 1], horizon_data[future_t, ee, 2], horizon_data[future_t, ee, 3], future_t])
                            break
                    else:
                        # If no contact (1) was found, add a default value (e.g., last state, max time step)
                        last_state = horizon_data[-1, ee]
                        output_list.extend([last_state[0], last_state[1], last_state[2], last_state[3], num_time_steps])
                    break

        return output_list

    def express_contact_plan_in_consistant_frame(self,
                                            q : np.ndarray,
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
        # X = [x_W, y_W, z_B] (all unknown)
        # A = [[1,0,0], [0,1,0], [0,0,0]] - B_R_W[:, -1]
        # B = W_R_B . ([x_B, y_B, 0].T - W_p_B) - [0, 0, z_W].T
        # (W_p_B = [0., 0., 0.].T as the contact plan is computed with the base at the origin)
        # Then X = A^{-1}.B
    
        # Reshape to process all positions at once
        cnt_plan_p = cnt_plan_pos.reshape(-1, 3).copy()

        # Rotation matrix W_R_B from world to base
        W_R_B = pinocchio.Quaternion(q[3:7]).matrix()  # Rotation matrix from base to world
        W_p_B = q[:3]  # Translation vector from base to world
        
        # Analytical form of the inverse of A
        A_inv = np.diag([1., 1., 1. / W_R_B[-1, -1]])
        A_inv[-1, :2] = - W_R_B[:2, -1] / W_R_B[-1, -1]

        # Prepare the contact positions for vectorized operation
        p_B = cnt_plan_p.copy()
        p_B[:, -1] = 0.  # Set z_B to 0 for all points

        # Compute B for all contact points in one operation
        B = W_R_B @ p_B.T  # Apply rotation to the base frame points
        B = B.T  # Transpose to get shape [N, 3]
        B[:, -1] -= cnt_plan_p[:, -1]  # Subtract z_W from the last coordinate of B

        # Compute X for all positions at once
        X = (A_inv @ B.T).T

        # Apply the final transformations based on the base_frame flag
        if base_frame:
            cnt_plan_p[:, -1] = X[:, -1]  # Update z_B in base frame
        else:
            cnt_plan_p[:, :-1] = X[:, :-1] + W_p_B[:-1]  # Update x_W and y_W in world frame

        # Reshape back to the original shape [H, 4, 3]
        cnt_plan_p = cnt_plan_p.reshape(-1, 4, 3)

        return cnt_plan_p