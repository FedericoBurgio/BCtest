import copy
import numpy as np
import os
from typing import Any, Dict, Callable
#import tensorflow as tf
import torch
import pickle
import time
import mujoco
import pinocchio as pin
from pinocchio.utils import zero
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
import time

gaits = [trot, jump]

from applyPert import apply_perturbation

def rec(comb, q0, v0, cntBools, sim_time, datasetName):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    q0, v0 = apply_perturbation(q0, v0, cntBools, robot)
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    recorder = Recorders.DataRecorderPD(controller)
    simulator = Simulator(robot, controller, recorder)
    #robot.reset(q0, v0)
    #FR foot 26 RL foot 42 RR foot 54
    #fl 13 fr 25 rl 41 rr 53 "foot geometries"
    #sim_time = 1.8  # seconds
    simulator.run(
        simulation_time=sim_time,
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )
    
    if not simulator.controller.diverged:
        simulator.data_recorder.save_data_hdf5(datasetName)
        return True
    else:
        return False

def recPert(comb, noStep, sim_time, datasetName):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    recorder = Recorders.DataRecorderNominal(controller)
    simulator = Simulator(robot, controller, recorder)
    #breakpoint()
    
    simulator.run( ##NOMINAL
        simulation_time=0.6,#.6 = 1 cycle
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )
    Step_ = 600//noStep
    #Step_ = 50
    qNom, vNom, cntBools = recorder.getNominal(Step_)
    
    del cfg
    del robot
    del controller
    del recorder
    del simulator

    i = 0
    kk=0
    try:
        data = np.load("progress.npz")
        i = data['i']
    except FileNotFoundError:
        i = 0  

    while i < len(qNom): 
        kk+=1
        if kk > 5 : break
        print(i,"/",len(cntBools))
        print(cntBools[i])
        if rec(comb, qNom[i], vNom[i], cntBools[i], sim_time, datasetName):    
            i += 1
            np.savez("progress.npz", i=i)
        print(i, "/", len(qNom))
        
    if i == len(qNom):
        i = 0
        np.savez("progress.npz", i=i)
        return True
    return False

def recPertLive(comb):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    recorder = Recorders.DataRecorderNominal(controller)
    simulator = Simulator(robot, controller, recorder)
   
    sim_time = 4.2# seconds
    simulator.run(
        simulation_time=sim_time,
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )
    
    qNom, vNom, cntBools = recorder.getNominal2(700)
    
    qNom0 = np.zeros_like(qNom)
    vNom0 = np.zeros_like(vNom)
    
    
    del cfg
    del robot
    del controller
    del recorder
    del simulator

    
    for i in range(5):
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
                *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                rotor_inertia=cfg.rotor_inertia,
                gear_ratio=cfg.gear_ratio,
                )
        for i in range(len(qNom)):
            qNom0[i], vNom0[i] = apply_perturbation(qNom[i], vNom[i], cntBools[i], robot)
            robot.reset()
        controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        recorder = Recorders.DataRecorderPD(controller)
        simulator = Simulator(robot, controller, recorder)
        simulator.run(
            simulation_time=sim_time,
            use_viewer=0,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb = comb,
            pertStep = 700,
            pertNomqs = qNom0,
            pertNomvs = vNom0,
            
            
        )
        
        if not simulator.controller.diverged:
            simulator.data_recorder.save_data_hdf5("pertTest")
            return True
            
        else:
            del cfg
            del robot
            del controller
            del recorder
            del simulator
    return False
                        
def SimManager(mode = - 1, dataset = "", ep = ""):
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
        # if dataset == "" or ep == "":
        #     dataset = "291532"
        #     ep = "final" #"ep120"
         
        #data = "251510"
        #ep = "130" 
        
        path = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth"
        NNprop = np.load("/home/atari_ws/project/models/" + dataset + "/NNprop.npz")
        controller = Controllers.TrainedControllerPD(robot.pin, state_size=NNprop['s_size'], action_size=NNprop['a_size'], hL=NNprop['h_layers'],
                                                     replanning_time=0.05, sim_opt_lag=False,
                                                     datasetPath = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth")
        # controller = Controllers.TrainedControllerPD(robot.pin, state_size=55, action_size=12, 
        #                                              replanning_time=0.05, sim_opt_lag=False,
        # #                                              datasetPath = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth")
        v_des, w_des = np.array([0.3, 0., 0.]), 0.0
        controller.set_command(v_des, w_des)
        # # Set gait
        #sel_gait = jump
        controller.set_gait_params(trot)  # Choose between trot, jump and bound
        # controller.gait_index = gaits.index(sel_gait)
        simulator = Simulator(robot, controller)
        
        # Visualize contact locations
        # visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        
        sim_time = 25#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb = [[0,0.3,0.,0]]
        )
        
    elif mode==2: 
        print("pinocchio")
        print("\n\n")
        robot.pin.info()
        print("\n\n")  
        print("mujoco")
        print("\n\n") 
        
        robot.mj.info()
        
        ###### Controller
        controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        
        # Set command
        #v_des, w_des = np.array([0.3, 0., 0.]), 0
        
        v_des, w_des = np.array([0, 0.3, 0.]), 0.0
      
        
        controller.set_command(v_des, w_des)
      
        # Set gait
        
        controller.set_gait_params(trot)  # Choose between trot, jump and bound
        simulator = Simulator(robot, controller) 
        
        sim_time = 4.2#s
        simulator.run(
            simulation_time=sim_time,
            use_viewer=True,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb = [[1,0.3,0.3,0.1]]
            )

def Record_with_pert(noStep, sim_time, datasetName):
    import itertools
    
    gaitsI = np.array([0, 1])
    vspacex = np.arange(-0.1, 0.6, 0.1)
    vspacey = np.arange(-0.4, 0.5, 0.1)
    wspace = np.arange(-0.07, 0.14, 0.07)
        
    from LHS import SampledVelSpace
    comb = SampledVelSpace(gaitsI, vspacex, vspacey, wspace, 40, seed_=99)
    try:
        data = np.load("progressJ.npz")
        j = data['j']        
    except FileNotFoundError:
        j= 0  
    
    if j < len(comb):
        print("j: ", j, "comb[j]: ",comb[j])
        if recPert([comb[j]], noStep, sim_time, datasetName):
            j = j + 1 
            np.savez("progressJ.npz", j=j)
        else:
            np.savez("progressJ.npz", j=j)
    
def replay(dataset, ep, comb = [[0,0.3,0,0]], sim_time = 10):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    path = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth"
    NNprop = np.load("/home/atari_ws/project/models/" + dataset + "/NNprop.npz")
    controller = Controllers.TrainedControllerPD(robot.pin, state_size=NNprop['s_size'], action_size=NNprop['a_size'], hL=NNprop['h_layers'],
                                                    replanning_time=0.05, sim_opt_lag=False,
                                                    datasetPath = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth")
    simulator = Simulator(robot, controller)
    
    simulator.run(
        simulation_time = sim_time,
        viewer=True,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )

if __name__ == "__main__":
    replay("062101", "final", comb = [[0,0.2,0,0],[1,0.4,0,0]], sim_time=3)
    # Record_with_pert(6,             #num samples per cycle 
    #                  1.2,            #pertubed episode duration
    #                 "boi000")       #saved dataset name
    #recPert([[0,0.3,0,0]])        
    #recPertLive([[0,0.3,0,0]])
    #recPert([[0,0.1,0,0],[0,0.3,0,0]])
    #SimManager(2)