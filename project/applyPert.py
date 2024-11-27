import numpy as np 
import pinocchio as pin 
from pinocchio.utils import zero

def apply_perturbation(q: np.ndarray, v: np.ndarray, 
                       cntBools: np.ndarray, pin_wrapper, gait=-1,
                       seed=None):
    # Extract the model and data from the wrapper
    pin_model = pin_wrapper.pin.model
    data = pin_wrapper.pin.data

    pin_wrapper.reset(q, v)
    pin.computeJointJacobians(pin_model, data, q)
    pin.forwardKinematics(pin_model, data, q, v)
    pin.updateFramePlacements(pin_model, data)
    
    nq = pin_model.nq  # Number of configuration variables
    nv = pin_model.nv  # Number of velocity variables

    # Frame IDs of the end-effectors
    EE_frames_all = np.array([14, 26, 42, 54], dtype=int)
    EE_frames = EE_frames_all[cntBools == 1]  # Feet in contact
    
    cnt_jac = np.zeros((3*len(EE_frames), nv))

    # Define rotate_jacobian function
    def rotate_jacobian(jac, index):
        world_R_joint = pin.SE3(data.oMf[index].rotation, pin.utils.zero(3))
        return world_R_joint.action @ jac

    # Compute contact Jacobians
    for ee_cnt, frame_id in enumerate(EE_frames):
        jac = pin.getFrameJacobian(pin_model, data, int(frame_id), pin.ReferenceFrame.LOCAL)
        cnt_jac[3*ee_cnt:3*(ee_cnt+1), :] = rotate_jacobian(jac, int(frame_id))[0:3, :]

    # Determine swing feet frames
    swing_feet_frames = EE_frames_all[cntBools == 0]  # Feet not in contact

    while True:
        # Sample perturbations
        if gait == 0:
            perturbation_pos = np.concatenate(([0,0,np.random.uniform(-.2,.2)],
                #np.random.normal(0, 0.18, 3),  # Base position
                np.random.uniform(-.2, 0.2, 3),  # Base orientation
                np.random.uniform(-.4, 0.4, nv - 6)  # Joint positions
            ))
            perturbation_vel = np.random.normal(0, 0.3, nv)
        elif gait == 1:
            perturbation_pos = np.concatenate((
                [0,0,0],  # Base position
                np.random.uniform(-0.2, 0.2, 3),   # Base orientation
                np.random.uniform(-.25, 0.25, nv - 6)  # Joint positions
            ))
            perturbation_vel = np.random.uniform(-0.15, 0.15, nv)
        # else:
        #     # Default perturbation if gait not specified
        #     perturbation_pos = np.random.normal(0, 0.05, nv)
        #     perturbation_vel = np.random.normal(0, 0.1, nv)

        if EE_frames.size == 0:
            random_pos_vec = perturbation_pos
            random_vel_vec = perturbation_vel
        else:
            # Compute projection matrix
            P = np.eye(nv) - np.linalg.pinv(cnt_jac) @ cnt_jac

            random_pos_vec = P @ perturbation_pos
            random_vel_vec = P @ perturbation_vel

        # Add perturbations to the state
        v0_ = v + random_vel_vec
        q0_ = pin.integrate(pin_model, q, random_pos_vec)
        
        # Update the robot's state and compute kinematics
        pin_wrapper.reset(q0_, v0_)
        pin.forwardKinematics(pin_model, data, q0_, v0_)
        pin.updateFramePlacements(pin_model, data)
        
        # Check if any swing foot is below ground
        swing_feet_above_ground = True
        for frame_id in swing_feet_frames:
            if data.oMf[int(frame_id)].translation[2] < 0.0:
                swing_feet_above_ground = False
                break  # No need to check further if one is below ground
        
        if swing_feet_above_ground:
            print("All swing feet are above ground.")
            break  # Exit the loop when all swing feet are above ground
        else:
            print("Resampling perturbations...")
            continue  # Resample perturbations

    # Final reset to the perturbed state
    pin_wrapper.reset(q0_, v0_)
    return q0_, v0_

