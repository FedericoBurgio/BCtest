import numpy as np
import pinocchio

def detect_contact_sequences(horizon_data): # in the sequence t0:1 t1:1 0 0 0 t5:1 find the next 1 and store the state and the number of steps taken to get there
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

def detect_contact_steps(horizon_data):#returns current pos if in contact, next pos if not in contactn
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

def detect_contact_steps2(horizon_data): #correzione di michal
    # Returns the whole predicted state when any boolean changes and the number of time steps it took to get there
    num_time_steps = horizon_data.shape[0]
    num_end_effectors = horizon_data.shape[1]

    initial_booleans = horizon_data[0, :, 0]  # Initial contact states for all end effectors

    for t in range(1, num_time_steps):
        current_booleans = horizon_data[t, :, 0]  # Contact states at time step t

        if not (current_booleans == initial_booleans).all():
            # A change in boolean state detected for at least one end effector
            return horizon_data[t], t  # Return the full state at time step t and the step index

    # If no change in boolean state occurs within the horizon, return the last state and total steps
    return horizon_data[-1], num_time_steps

def detect_contact_steps3(horizon_data, q):
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
            cnt_plan_pos_modified = express_contact_plan_in_consistant_frame(q, cnt_plan_pos, base_frame=True)
            
            # Overwrite the original contact plan positions in horizon_data with the modified ones
            horizon_data[t, :, 1:] = cnt_plan_pos_modified  # Only modify the part without the boolean
            
            # Return the modified full state at time step t and the step index
            return horizon_data[t], t

    # If no change in boolean state occurs within the horizon, return the last state and total steps

    cnt_plan_pos = horizon_data[-1, :, 1:]  # Extract last contact plan positions
    cnt_plan_pos_modified = express_contact_plan_in_consistant_frame(q, cnt_plan_pos)  # Modify last positions
    horizon_data[-1, :, 1:] = cnt_plan_pos_modified  # Overwrite last contact plan with modified one

    return horizon_data[-1], num_time_steps

def detect_contact_steps4(horizon_data, q):

    # Returns the whole predicted state when any boolean changes and the number of time steps it took to get there
    num_time_steps = horizon_data.shape[0]
    num_end_effectors = horizon_data.shape[1]
    positions_dim = horizon_data.shape[2] - 1  # Number of position columns after the boolean

    initial_positions = horizon_data[0, :, 1:]  # Positions at time t=0

    for t in range(1, num_time_steps):
        prev_booleans = horizon_data[t - 1, :, 0]  # Booleans at time t-1
        current_booleans = horizon_data[t, :, 0]   # Booleans at time t

        if not (current_booleans == prev_booleans).all():
            # A change in boolean state detected for at least one end effector

            # Initialize an array to hold the contact plan positions
            cnt_plan_pos = np.zeros((num_end_effectors, positions_dim))
            #br = False
            for ee in range(num_end_effectors):
                #   if prev_booleans[ee] == 0 and current_booleans[ee] == 1: br = True
                if prev_booleans[ee] == 1 and current_booleans[ee] == 0:
                    # If the boolean changes from 1 to 0, use the initial position at time t=0
                    cnt_plan_pos[ee] = initial_positions[ee]
                else:
                    # Otherwise, use the position at time t
                    cnt_plan_pos[ee] = horizon_data[t, ee, 1:]

            # Modify the contact plan using express_contact_plan_in_consistent_frame
            cnt_plan_pos_modified = express_contact_plan_in_consistant_frame(q, cnt_plan_pos, base_frame=True)

            # Overwrite the original contact plan positions in horizon_data with the modified ones
            horizon_data[t, :, 1:] = cnt_plan_pos_modified  # Only modify the part without the boolean

            # Return the modified full state at time step t and the step index
            #if br: breakpoint()
            return horizon_data[t], t

    # If no change in boolean state occurs within the horizon, return the last state and total steps

    cnt_plan_pos = horizon_data[-1, :, 1:]  # Extract last contact plan positions
    cnt_plan_pos_modified = express_contact_plan_in_consistant_frame(q, cnt_plan_pos)  # Modify last positions
    horizon_data[-1, :, 1:] = cnt_plan_pos_modified  # Overwrite last contact plan with modified one

    return horizon_data[-1], num_time_steps


def express_contact_plan_in_consistant_frame(q: np.ndarray,
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