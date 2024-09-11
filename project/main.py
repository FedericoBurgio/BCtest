import copy
import numpy as np
import os
from typing import Any, Dict, Callable
#import tensorflow as tf
import torch
import pickle
import time
import mujoco
from mujoco import viewer
from mj_pin_wrapper.abstract.robot import AbstractRobotWrapper
from mj_pin_wrapper.mj_robot import MJQuadRobotWrapper
from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.abstract.data_recorder import DataRecorderAbstract


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

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_size)
        self.dropout = nn.Dropout(0.05)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        action_logits = self.fc4(x)
        return action_logits
    

import Controllers
import Recorders
 
if __name__ == "__main__":
    
    # Prompt the user to enter the mode number
    mode = input("\nWhich mode do you want to use?\n0: recording\n1: testing trained\n2: MPC\n")

    try:
        # Convert input to an integer
        mode = int(mode)
    except ValueError:
        print("Invalid input. Please enter a valid number.\n")
        
    
    import random
    
    std_deviation = 0.0655# initial state randomness
    
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    
    if mode==0: #recording  
    ##### Robot model
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
        simulator = Simulator(robot.mj, controller, Recorders.DataRecorderPD(controller)) 
        
        sim_time = 1#s
        simulator.run(
            simulation_time=sim_time,
            use_viewer=False,
            real_time=False,
            visual_callback_fn=None,)
        if not simulator.controller.diverged:
            simulator.data_recorder.save_data("DatasetPD2TEST.npz")
            
    elif mode==1:

        controller = Controllers.TrainedControllerPD(robot.pin, replanning_time=0.05, sim_opt_lag=False,datasetPath="/home/atari_ws/project/111227/best_policy_final.pth")
        simulator = Simulator(robot.mj, controller)
        
        # Visualize contact locations
        # visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        
        sim_time = 15#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
        )
        
    ###
    elif mode==2: 
        robot.pin.info()
        robot.mj.info()
        
        ###### Controller
        controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        
        # Set command
        v_des, w_des = np.array([0.3, 0., 0.]), 0
        controller.set_command(v_des, w_des)
        # Set gait
        
        controller.set_gait_params(trot)  # Choose between trot, jump and bound
        simulator = Simulator(robot.mj, controller) 
        
        sim_time = 20#s
        simulator.run(
            simulation_time=sim_time,
            use_viewer=True,
            real_time=False,
            visual_callback_fn=None,)
    ###
        
    elif mode == 999:
        5 