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
       
    
        # Select gait based on success, wrapping around using modulo
        #sel_gait = gaits[success % len(gaits)]
        
        v_des = np.zeros(3)
        w_des = 0
        vspace = np.linspace(-0.3, 0.3, 5) #0.15 step
        #i = len(vspace)
        #j = len(vspace)
        #selGait = len(gaits)
        i=0
        j=0
        k=0
   
        # Try to load progress from 'progress.npz'
        try:
            data = np.load("progress.npz")
            i, j, k = data['i'], data['j'], data['k']
        except FileNotFoundError:
            i, j, k = 0, 0, 0  
        print(i)
        print(j)
        print(k)
        print("_____________")
        if i == 1: vspace = np.linspace(0.05, 0.6, 5)
        kk=0
        # for i in range(i, len(gaits)):
        #     for j in range(j, len(vspace)):
        #         for k in range(k, len(vspace)):
        #             sel_gait = gaits[i]
        #             v_des[0] = vspace[j]
        #             v_des[1] = vspace[k]
        #             w_des = 0
        #             # Set command and gait parameters
        #             controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        #             controller.set_command(v_des, w_des)
        klim = 10
        while i < len(gaits):
            while j < len(vspace):
                while k < len(vspace):
                    if kk > klim: break
                    # Reinitialize robot after each inner loop iteration
                    cfg = Go2Config
                    robot = MJPinQuadRobotWrapper(
                        *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                        rotor_inertia=cfg.rotor_inertia,
                        gear_ratio=cfg.gear_ratio,
                    )
                    print(i)
                    print(j)
                    print(k)
                    
                    sel_gait = gaits[i]
                    v_des[0] = vspace[j]
                    v_des[1] = vspace[k]
                    w_des = 0#np.random.uniform(-0.01, 0.01)
                    
                    # Set command and gait parameters
                    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
                    controller.set_command(v_des, w_des)
                    controller.set_gait_params(sel_gait)
                    
                    recorder = Recorders.DataRecorderPD(controller)
                    recorder.gait_index = i
                    
                    simulator = Simulator(robot.mj, controller, recorder)
                    
                    sim_time = 2.5  # seconds
                    simulator.run(
                        simulation_time=sim_time,
                        use_viewer=0,
                        real_time=False,
                        visual_callback_fn=None,
                        randomize=True
                    )
                    
                    if not simulator.controller.diverged:
                        simulator.data_recorder.save_data("55s_noRND.npz")
                        k += 1
                        kk+=1
                    
                    # Increment k and save progress
                    np.savez("progress.npz", i=i, j=j, k=k)
                    
                    
                    if kk > klim: break
                    del robot
                    del controller
                    del simulator
                    del cfg
                # Increment j after finishing the inner k loop
                if kk > klim: break
                j += 1
                k = 0  # Reset k when starting a new j iteration
                
                # Save progress after updating j
                np.savez("progress.npz", i=i, j=j, k=k)
            
            # Increment i after finishing the j loop
            if kk > klim: break
            i += 1
            j = 0
            k = 0

            # Save progress after updating i
            np.savez("progress.npz", i=i, j=j, k=k)
        
        print("Finished recording data.")
        # try:
        #     os.remove("progress.npz")
        #     print("Progress file deleted.")
        # except FileNotFoundError:
        #     print("Progress file not found, nothing to delete.")
        

            
    elif mode==1:
        data = "021333"
        ep = "final" #"ep120" 
        #data = "251510"
        #ep = "130" 
        
        path = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth"
        NNprop = np.load("/home/atari_ws/project/models/" + data + "/NNprop.npz")
        controller = Controllers.TrainedControllerPD(robot.pin, state_size=NNprop['s_size'], action_size=NNprop['a_size'], hL=NNprop['h_layers'],
                                                     replanning_time=0.05, sim_opt_lag=False,
                                                     datasetPath = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth")
        # controller = Controllers.TrainedControllerPD(robot.pin, state_size=55, action_size=12, 
        #                                              replanning_time=0.05, sim_opt_lag=False,
        #                                              datasetPath = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth")
        v_des, w_des = np.array([0.3, 0., 0.]), 0.0
        controller.set_command(v_des, w_des)
        # Set gait
        sel_gait = jump
        controller.set_gait_params(sel_gait)  # Choose between trot, jump and bound
        controller.gait_index = gaits.index(sel_gait)
        simulator = Simulator(robot.mj, controller)
        
        # Visualize contact locations
        # visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        
        sim_time = 30#s
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
        
        v_des, w_des = np.array([-0.3, 0., 0.]), 0.0
      
        
        controller.set_command(v_des, w_des)
      
        # Set gait
        
        controller.set_gait_params(jump)  # Choose between trot, jump and bound
        simulator = Simulator(robot.mj, controller) 
        
        sim_time = 20#s
        simulator.run(
            simulation_time=sim_time,
            use_viewer=True,
            real_time=False,
            visual_callback_fn=None,
            randomize=False)
    ###
        
    elif mode == 999:
        5 
    


if __name__ == "__main__":
    SimManager()
    