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


from mj_pin_wrapper.abstract.controller import ControllerAbstract
from mj_pin_wrapper.sim_env.utils import RobotModelLoader
from mj_pin_wrapper.mj_pin_robot import MJPinQuadRobotWrapper
from mj_pin_wrapper.simulator import Simulator
from utils.visuals import desired_contact_locations_callback
from utils.config import Go2Config

import torch
import torch.nn as nn

import nets
import Controllers
import Recorders

def SimManager(mode = - 1):
    # Prompt the user to enter the mode number
    if mode == -1:
        mode = input("\nWhich mode do you want to use?\n0: recording\n1: testing trained\n2: MPC\n")

        try:
            # Convert input to an integer
            mode = int(mode)
        except ValueError:
            print("Invalid input. Please enter a valid number.\n")
            
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    
    gaits = [trot, jump]
    if mode == 0:
        success = 0
        while success < 16:
            controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
            
            # Set command
            
            # Select gait based on success, wrapping around using modulo
            sel_gait = gaits[success % len(gaits)]
            
            v_des = np.zeros(3)
            v_des[0] = np.random.uniform(-0.5, 0.5)
            v_des[1] = np.random.uniform(-0.25, 0.25)
            w_des = np.random.uniform(-0.055, 0.055)
            
            if sel_gait == jump: 
                controller.set_command(v_des * 0.5, w_des * 0.4)
            else:
                controller.set_command(v_des, w_des)
            
            # Set gait parameters and create simulator
            controller.set_gait_params(sel_gait)  # Choose between trot, jump, etc.
            recorder = Recorders.DataRecorderPD(controller)
            recorder.gait_index = gaits.index(sel_gait)
            simulator = Simulator(robot.mj, controller, recorder)
            
            sim_time = 8  # seconds
            simulator.run(
                simulation_time=sim_time,
                use_viewer=0,
                real_time=False,
                visual_callback_fn=None,
                randomize=True)
            
            if not simulator.controller.diverged:
                simulator.data_recorder.save_data("v_w_gaits-jump-trot.npz")
                success += 1
            
            print("Successful:", success)

            # Reinitialize robot
            cfg = Go2Config
            robot = MJPinQuadRobotWrapper(
                *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                rotor_inertia=cfg.rotor_inertia,
                gear_ratio=cfg.gear_ratio,
            )
            
    elif mode==1:
        data = "291101"
        ep = "ep250" #"ep120" 
        #data = "251510"
        #ep = "130" 
        
        path = "/home/atari_ws/project/models/" + data + "/best_policy_" + ep + ".pth"
        controller = Controllers.TrainedControllerPD(robot.pin, replanning_time=0.05, sim_opt_lag=False,
                                                     datasetPath = path)
        
        v_des, w_des = np.array([-0.2, -0.2, 0.]), 0.
        controller.set_command(v_des, w_des)
        # Set gait
        sel_gait = trot
        controller.set_gait_params(sel_gait)  # Choose between trot, jump and bound
        controller.gait_index = gaits.index(sel_gait)
        simulator = Simulator(robot.mj, controller)
        
        # Visualize contact locations
        # visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        
        sim_time = 35#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
            randomize=False
        )
        
    ###
    elif mode==2: 
        robot.pin.info()
        robot.mj.info()
        
        ###### Controller
        controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        
        # Set command
        #v_des, w_des = np.array([0.3, 0., 0.]), 0
        
        v_des = np.zeros(3)
        v_des[0] = np.random.uniform(-0.5, 0.5)
        v_des[1] = np.random.uniform(-0.25, 0.25)
        w_des = np.random.uniform(-0.055, 0.055)
        controller.set_command(v_des, w_des)
        # Set gait
        
        controller.set_gait_params(trot)  # Choose between trot, jump and bound
        simulator = Simulator(robot.mj, controller) 
        
        sim_time = 20#s
        simulator.run(
            simulation_time=sim_time,
            use_viewer=True,
            real_time=False,
            visual_callback_fn=None,
            randomize=True)
    ###
        
    elif mode == 999:
        5 
    


if __name__ == "__main__":
    SimManager()
    