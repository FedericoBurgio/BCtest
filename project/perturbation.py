from typing import Any, Callable
import mujoco
from mujoco import viewer
import time
import numpy as np

from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract

import mujoco
import nets
import Controllers
import Recorders
import pinocchio as pin


from mpc_controller.motions.cyclic.go2_trot import trot
from mpc_controller.motions.cyclic.go2_jump import jump
from mpc_controller.motions.cyclic.go2_bound import bound
from mpc_controller.bicon_mpc import BiConMPC

from pinocchio.utils import zero
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

from applyPert import apply_perturbation

def apply_perturbationBUP(q: np.ndarray, v: np.ndarray, cntBools: np.ndarray, pin_wrapper,seed=None):

    # Extract the model and data from the wrapper
    pin_model = pin_wrapper.pin.model
    data = pin_wrapper.pin.data
    
    pin_wrapper.reset(q, v)
    pin.computeJointJacobians(pin_model, data, q)
    pin.forwardKinematics(pin_model, data, q, v)
    
    nq = pin_model.nq  # Number of configuration variables
    nv = pin_model.nv  # Number of velocity variables

    # Frame IDs of the end-effectors (modify these IDs according to your robot's end-effectors)
    EE_frames_all = np.array([14, 26, 42, 54], dtype=int)  # Ensure dtype is int
    EE_frames = EE_frames_all[cntBools == 1]
    
    cnt_jac = []
    cnt_jac_dot = []
    
    def rotate_jacobian(jac, index):
        world_R_joint = pin.SE3(data.oMf[index].rotation, zero(3))
        return world_R_joint.action @ jac

    # Ensure data is up-to-date
    pin.forwardKinematics(pin_model, data, q, v)
    pin.updateFramePlacements(pin_model, data)

    for ee_cnt in range(len(EE_frames)):
        frame_id = int(EE_frames[ee_cnt])

        # Get the Jacobian and its time derivative
        jac = pin.getFrameJacobian(pin_model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        jac_dot = pin.getFrameJacobianTimeVariation(pin_model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Rotate Jacobians
        rotated_jac = rotate_jacobian(jac, frame_id)
        rotated_jac_dot = rotate_jacobian(jac_dot, frame_id)

        # Extract linear part (first 3 rows)
        cnt_jac.append(rotated_jac[:3, :])
        cnt_jac_dot.append(rotated_jac_dot[:3, :])

    if len(cnt_jac) > 0:
        cnt_jac = np.vstack(cnt_jac)
        cnt_jac_dot = np.vstack(cnt_jac_dot)
    else:
        cnt_jac = np.zeros((0, nv))
        cnt_jac_dot = np.zeros((0, nv))

    # Build the contact constraint matrix A_c
    num_constraints = cnt_jac.shape[0]
    zeros_nv = np.zeros((num_constraints, nv))

    A_c_upper = np.hstack([cnt_jac, zeros_nv])     # [J_c, 0]
    A_c_lower = np.hstack([cnt_jac_dot, cnt_jac])  # [\dot{J}_c, J_c]

    A_c = np.vstack([A_c_upper, A_c_lower])        # Shape: (2 * num_constraints, 2 * nv)
    np.random.seed(seed=seed)
    # Generate random perturbations
    delta_q = np.concatenate((np.random.normal(0, .16, 3),\
                        np.random.normal(0, .14, 3), \
                        np.random.normal(0, 0.46, nv-6)))
    delta_v = np.random.normal(0, 0.3, nv)

    delta_qv = np.concatenate([delta_q, delta_v])
    
    #NOTA BENE
    #delta_qv = np.zeros(2*nv)
    # Compute the projection matrix
    if A_c.shape[0] > 0:
        A_c_pinv = np.linalg.pinv(A_c)
        P = np.eye(2 * nv) - A_c_pinv @ A_c
        delta_qv_c = P @ delta_qv
    else:
        delta_qv_c = delta_qv

    # Extract the projected perturbations
    delta_q_c = delta_qv_c[:nv]
    delta_v_c = delta_qv_c[nv:]

    # Apply the perturbations
    v0_ = v + delta_v_c
    q0_ = pin.integrate(pin_model, q, delta_q_c)

    # Ensure the quaternion remains normalized
    quat = q0_[3:7]
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-6:
        q0_[3:7] = quat / quat_norm
    else:
        q0_[3:7] = q[3:7]  # Reset to original if normalization fails

    # Check if any swing foot is below the ground
    pin.forwardKinematics(pin_model, data, q0_)
    pin.updateFramePlacements(pin_model, data)
    ee_below_ground = []
    for e in range(len(EE_frames_all)):
            frame_id = int(EE_frames_all[e])
            if data.oMf[frame_id].translation[2] < 0.0001:
                ee_below_ground.append(1)

    # Repeat perturbation if any swing foot is below ground
    if len(ee_below_ground) > 0:
        return apply_perturbation(q, v, cntBools, pin_wrapper)
    pin_wrapper.reset(q0_, v0_)
    return q0_, v0_

def apply_perturbationOLD(q: np.ndarray, v: np.ndarray, cntBools: np.ndarray, pin_wrapper, seed = None):
    # Extract the model and data from the wrapper
    pin_model = pin_wrapper.pin.model
    data = pin_wrapper.pin.data

    pin_wrapper.reset(q, v)
    pin.computeJointJacobians(pin_model, data, q)
    pin.forwardKinematics(pin_model, data, q, v)
    
    nq = pin_model.nq  # Number of configuration variables (19)
    nv = pin_model.nv  # Number of velocity variables (18)

    # Frame IDs of the end-effectors (modify these IDs according to your robot's end-effectors)
    EE_frames_all = np.array([14, 26, 42, 54], dtype=int)  # Ensure dtype is int
    EE_frames = EE_frames_all[cntBools == 1]
    
    cnt_jac = np.zeros((3*len(EE_frames), nv))
    cnt_jac_dot = np.zeros((3*len(EE_frames), nv))
    
    def rotate_jacobian(jac, index):
        world_R_joint = pin.SE3(data.oMf[index].rotation, zero(3))
        #return world_R_joint.action @ jac
        return jac
    for ee_cnt in range(len(EE_frames)):
        
        jac = pin.getFrameJacobian(pin_model,\
            data,\
            int(EE_frames[ee_cnt]),\
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        cnt_jac[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(jac,int(EE_frames[ee_cnt]))[0:3,]
        jac_dot = pin.getFrameJacobianTimeVariation(pin_model,\
            data,\
            int(EE_frames[ee_cnt]),\
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        cnt_jac_dot[3*ee_cnt:3*(ee_cnt+1),] = rotate_jacobian(jac_dot,int(EE_frames[ee_cnt]))[0:3,]

    min_ee_height=.0
    while min_ee_height >= 0:
        np.random.seed(seed=seed)
        perturbation_pos = np.concatenate((np.random.normal(0, .16, 3),\
                        np.random.normal(0, .14, 3), \
                        np.random.normal(0, 0.46, nv-6)))
        
        perturbation_vel = np.random.normal(0, 0.3, nv)
        if EE_frames.size == 0:
            random_pos_vec = perturbation_pos
            random_vel_vec = perturbation_vel
        else:
            random_pos_vec = (np.identity(nv) - np.linalg.pinv(cnt_jac)@\
                        cnt_jac) @ perturbation_pos
            jac_vel = cnt_jac_dot * perturbation_pos + cnt_jac * perturbation_vel
            random_vel_vec = (np.identity(nv) - np.linalg.pinv(jac_vel)@\
                        jac_vel) @ perturbation_pos

        ### add perturbation to nominal trajectory
        v0_ = v + random_vel_vec
        q0_ = pin.integrate(pin_model, \
            q, random_pos_vec) #random_pos_vec, vel * dt ,where dt is implicit  
        ###NOTA BENE
        # v0_ = v
        # q0_ = q 
        ### check if the swing foot is below the ground
        pin.forwardKinematics(pin_model, data, q0_, v0_)
        pin_wrapper.reset(q0_, v0_)
        pin.framesForwardKinematics(pin_model, data, q0_)
        pin.updateFramePlacements(pin_model, data)
        ee_below_ground = []
        for e in range(len(EE_frames_all)):
            frame_id = int(EE_frames_all[e])
            if data.oMf[frame_id].translation[2] < 0.0001:
                ee_below_ground.append(1)
        if len(ee_below_ground) == 0: # ee_below_ground==[]
            min_ee_height = -1.
    pin_wrapper.reset(q0_, v0_)
        
    return q0_, v0_

cfg = Go2Config
robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )
controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
recorder = Recorders.DataRecorderNominal(controller)
simulator = Simulator(robot, controller, recorder)

comb = [[1,0.3,0,0]]

sim_time = 0.5 # seconds
simulator.run(
    simulation_time=sim_time,
    use_viewer=0,
    real_time=False,
    visual_callback_fn=None,
    randomize=False,
    comb = comb
)

qNom, vNom, cntBools = recorder.getNominal(20)
#qNom, vNom, cntBools = recorder.getNominal(20) #stato 6 ha qualcosa che non va
# qNom=qNom[:12]
# vNom=vNom[:12]
# cntBools=cntBools[:12] 
  
del cfg
del robot
del controller
del recorder
del simulator

# Load your MuJoCo model
cfg = Go2Config
robot = MJPinQuadRobotWrapper(
        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
        rotor_inertia=cfg.rotor_inertia,
        gear_ratio=cfg.gear_ratio,
        )
data = robot.mj.data
# Viewer setup

robot.pin.info()
# Define your list of initial states as tuples (q, v)
# For example: initial_states = [(q1, v1), (q2, v2), ...]
# where each q and v is an np.array of appropriate shape for your model

    # Add more states as needed

# Loop through each initial state
for i in range(len(qNom)):
    print("State", i)   
    # Set the initial position and velocity
    robot.reset(qNom[i], vNom[i])
    with mujoco.viewer.launch_passive(robot.mj.model, robot.mj.data) as viewer:
        # Render the state in the viewer
        
        print("Nominal", cntBools[i])
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        
        viewer.sync()
        #viewer.render()

        # Optional: Pause for a moment to visualize the state
        input()
    
    apply_perturbation(qNom[i], vNom[i], cntBools[i], robot, gait=comb[0][0])    
    with mujoco.viewer.launch_passive(robot.mj.model, robot.mj.data) as viewer:
        # Render the state in the viewer
        
        print("OLD", cntBools[i])
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        
        viewer.sync()
        #viewer.render()

        # Optional: Pause for a moment to visualize the state
        input()
    # apply_perturbationBUP(qNom[i], vNom[i], cntBools[i], robot,seed=i*2)    
    # with mujoco.viewer.launch_passive(robot.mj.model, robot.mj.data) as viewer:
    #     # Render the state in the viewer
    #     print("NEW",cntBools[i])
    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    #     viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    #     viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    #     viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        
    #     viewer.sync()
    #     #viewer.render()

    #     # Optional: Pause for a moment to visualize the state
    #     input()

# Close the viewer when done
viewer.close()
