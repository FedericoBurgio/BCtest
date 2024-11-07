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

gaits = [trot, jump]

def recording(comb, sim_time = 4, mode = 0, fails = 0):
    success = True
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    
    #gaits = [trot, jump]
    for i in range(len(comb)):
        comb[i][-1]=comb[i][-1]*(1-fails/5)
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    # controller.set_command(v_des, w_des)
    # controller.set_gait_params(gaits[gait_index])
    
    recorder = Recorders.DataRecorderPD(controller)
   
    
    simulator = Simulator(robot.mj, controller, recorder)
    
    #sim_time = 4  # seconds
    simulator.run(
        simulation_time=sim_time,
        use_viewer=False,
        real_time=False,
        visual_callback_fn=None,
        randomize=True,
        mode = mode,
        comb = comb
    )
    
    if not simulator.controller.diverged:
        simulator.data_recorder.save_data_hdf5("73s_21ottSwitch2")#.h5 nel recorder 
        success = True
    else: 
        success = False
    del robot
    del cfg
    del controller
    del simulator
    del recorder
    return success
    # Increment k and save progress
        
def SimManager(mode = - 1):
    # Prompt the user to enter the mode number
   # mode = 1
    if mode == -1:
        #while mode != [0,1,2,10]:
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
    
    
    if mode == 10:
        import itertools

        vspacex = np.arange(-0.1, 0.7, 0.05) #8
        vspacey = np.arange(-0.4, 0.5, 0.05) #9
        wspace = np.arange(-0.06, 0.06, 0.01)#6
        xy_combinations = list(itertools.product(vspacex, vspacey, wspace))
        comb = [[v, x, y, w] for x, y, w in xy_combinations for v in range(len(gaits))]
        
        def alternating_order(arr): # non mi ricordo
            n = len(arr)
            result = []
            
            for i in range(n // 2):
                result.append(arr[i])        # Take from the beginning
                result.append(arr[n - i - 1]) # Take from the end
            
            # If there's an odd number of elements, add the middle one
            if n % 2 == 1:
                result.append(arr[n // 2])
            
            return result
        
        def rearrange_pairs(sequence): # fa in modo che cambi o gait o vdes, wdes
            n = len(sequence)
            result = []

            # Iterate through the list two elements at a time
            for i in range(0, n, 4):
                # Add the first two elements in the current chunk
                if i + 1 < n:
                    result.append(sequence[i])
                    result.append(sequence[i + 1])
                
                # Add the next two elements but swap them
                if i + 3 < n:
                    result.append(sequence[i + 3])
                    result.append(sequence[i + 2])
            
            return result
        
        comb = rearrange_pairs(comb)
        

        #Try to load progress from 'progress.npz'
        try:
            data = np.load("progress.npz")
            i = data['i']
        except FileNotFoundError:
            i = 0  
        
        fails = 0
        kk=0
        #breakpoint()
        
        while i < (len(comb)-3):
            if kk > 4: break
        
            print("_________________________________")
            print("recording ", i, "/", len(comb),"\n")

            # gait_index = comb[i][0] # index of the gait
            # #sel_gait = gaits[gait_index]    
            # v_des[0:2] = comb[i][1:3] 
            # w_des =  np.random.uniform(-0.03, 0.03)
            success = recording(comb[i:i+3], sim_time=2, fails=fails)
            
            if success:
                kk+=1
                i+=15
                fails = 0
                np.savez("progress.npz", i=i)
            else:
                fails +=1
                if fails > 3:
                    i+=1
                    fails = 0
                    print("Too many fails, skipping to next")
            
    if mode == 0:
        
        vspacex = np.arange(-0.1, 0.7, 0.1) #8
        vspacey = np.arange(-0.4, 0.5, 0.1) #9
        wspace = np.arange(-0.06, 0.06, 0.02)#6
        
        import itertools
        comb = list(itertools.product(np.arange(0,len(gaits)), vspacex, vspacey, wspace))
        
        #xy_combinations = list(itertools.product(vspacex, vspacey))
        #comb = [[v, x, y] for x, y in xy_combinations for v in range(len(gaits))]
        #breakpoint()
        def alternating_order(arr): # non mi ricordo # forse serve per alternare primo ultimo secondo penultimo terzo terzultimo ecc
            n = len(arr)
            result = []
            
            for i in range(n // 2):
                result.append(arr[i])         # Take from the beginning
                result.append(arr[n - i - 1]) # Take from the end
            
            # If there's an odd number of elements, add the middle one
            if n % 2 == 1:
                result.append(arr[n // 2])
            
            return result
        
        def rearrange_pairs(sequence): # fa in modo che cambi o gait o vdes, wdes
            n = len(sequence)
            result = []

            # Iterate through the list two elements at a time
            for i in range(0, n, 4):
                # Add the first two elements in the current chunk
                if i + 1 < n:
                    result.append(sequence[i])
                    result.append(sequence[i + 1])
                
                # Add the next two elements but swap them
                if i + 3 < n:
                    result.append(sequence[i + 3])
                    result.append(sequence[i + 2])
            
            return result
        
        fails = 0
        kk=0

        #Try to load progress from 'progress.npz'
        try:
            data = np.load("progress.npz")
            i = data['i']
        except FileNotFoundError:
            i = 0  
        
        while i < len(comb):
            if kk > 10: break
            print("_________________________________")
            print("recording ", i, "/", len(comb),"\n")

            success = recording([comb[i]], sim_time=4, fails=fails)
            
            if success:
                kk+=1
                i+=1
                fails = 0
                np.savez("progress.npz", i=i)
            else:
                fails +=1
                if fails > 3:
                    i+=1
                    fails = 0
                    print("Too many fails, skipping to next")

    
    # if mode == 0:
       
    
    #     # Select gait based on success, wrapping around using modulo
    #     #sel_gait = gaits[success % len(gaits)]
        
    #     v_des = np.zeros(3)
    #     w_des = 0
    #     vspacex = np.linspace(-0.15, 0.6, 7) 
    #     vspacey = np.linspace(-0.5, 0.5, 7)
    #     i=0
    #     j=0
    #     k=0
    #     kk=0
   
    #     #Try to load progress from 'progress.npz'
    #     try:
    #         data = np.load("progress.npz")
    #         i, j, k = data['i'], data['j'], data['k']
    #     except FileNotFoundError:
    #         i, j, k = 0, 0, 0  
    #     print(i)
    #     print(j)
    #     print(k)
    #     print("_____________")
    #     #if i == 1: vspace = np.linspace(-0.1, 0.6, 8)

    #     klim = 10
    #     while i < len(gaits):
    #         while j < len(vspacex):
    #             while k < len(vspacey):
    #                 if kk > klim: break
    #                 #
    #                 # Reinitialize robot after each inner loop iteration
    #                 cfg = Go2Config
    #                 robot = MJPinQuadRobotWrapper(
    #                         *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
    #                         rotor_inertia=cfg.rotor_inertia,
    #                         gear_ratio=cfg.gear_ratio,
    #                         )
                
    #                 print(i)
    #                 print(j)
    #                 print(k)
                    
    #                 sel_gait = gaits[i]
    #                 v_des[0] = vspacex[j] # x, direzione frontale
    #                 v_des[1] = vspacey[k] # y, direzione laterale
    #                 w_des = np.random.uniform(-0.025, 0.025)
    #                 #recording(v_des,w_des,gaits,gait_index = i)
    #                 # Set command and gait parameters
    #                 controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    #                 controller.set_command(v_des, w_des)
    #                 controller.set_gait_params(sel_gait)
                    
    #                 recorder = Recorders.DataRecorderPD(controller)
    #                 recorder.gait_index = i
                    
    #                 simulator = Simulator(robot.mj, controller, recorder)
                    
    #                 sim_time = 4  # seconds
    #                 simulator.run(
    #                     simulation_time=sim_time,
    #                     use_viewer=False,
    #                     real_time=False,
    #                     visual_callback_fn=None,
    #                     randomize=True
    #                 )
                    
    #                 if not simulator.controller.diverged:
    #                     simulator.data_recorder.save_data("73s_8ott.npz")
    #                     k += 1
    #                     kk += 1 
                    
    #                 # Increment k and save progress
    #                 np.savez("progress.npz", i=i, j=j, k=k)
                    
                    
    #                 if kk > klim: break
    #                 del robot
    #                 del controller
    #                 del simulator
    #                 del cfg
    #             # Increment j after finishing the inner k loop
    #             if kk > klim: break
    #             j += 1
    #             k = 0  # Reset k when starting a new j iteration
                
    #             # Save progress after updating j
    #             np.savez("progress.npz", i=i, j=j, k=k)
            
    #         # Increment i after finishing the j loop
    #         if kk > klim: break
    #         i += 1
    #         j = 0
    #         k = 0

    #         # Save progress after updating i
    #         np.savez("progress.npz", i=i, j=j, k=k)
        
    #     print("Finished recording data.")
    #     # try:
    #     #     os.remove("progress.npz")
    #     #     print("Progress file deleted.")
    #     # except FileNotFoundError:
    #     #     print("Progress file not found, nothing to delete.")
        

            
    elif mode==1:
        data = "151109" #"081529" #031816 ha funzionto, all'episodio 250 sembrava ancora avesse mmiglioramente in val loss, in geneerale ha fatto progressi quai ad ogni ep
        ep = "final" #"ep120" 
        #data = "251510"
        #ep = "130" 
        
        import itertools

        v_des = np.zeros(3)
        w_des = 0
        vspacex = np.linspace(-0.1, 0.6, 20) 
        vspacey = np.linspace(-0.1, 0.6, 20)
        xy_combinations = list(itertools.product(vspacex, vspacey))
        comb = [[v, x, y] for x, y in xy_combinations for v in range(len(gaits))]
        print(comb) #[[index, v_des x, v_des y], ..., []] #list
        
        path = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth"
        NNprop = np.load("/home/atari_ws/project/models/" + data + "/NNprop.npz")
        controller = Controllers.TrainedControllerPD(robot.pin, state_size=NNprop['s_size'], action_size=NNprop['a_size'], hL=NNprop['h_layers'],
                                                     replanning_time=0.05, sim_opt_lag=False,
                                                     datasetPath = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth")
        # controller = Controllers.TrainedControllerPD(robot.pin, state_size=55, action_size=12, 
        #                                              replanning_time=0.05, sim_opt_lag=False,
        #                                              datasetPath = "/home/atari_ws/project/models/" + data + "/best_policy_ep" + ep + ".pth")
        v_des, w_des = np.array([0.0, 0.3, 0.3]), 0.
        controller.set_command(v_des, w_des)
        # Set gait
        sel_gait = jump
        controller.set_gait_params(sel_gait)  # Choose between trot, jump and bound
        controller.gait_index = gaits.index(sel_gait)
        simulator = Simulator(robot.mj, controller)
        
        # Visualize contact locations
        #visual_callback = (lambda viewer, step, q, v, data :
        #     desired_contact_locations_callback(viewer, step, q, v, data, controller))
        # Run simulation
        vspacex = np.arange(-0.1, 0.7, 0.05) #8
        vspacey = np.arange(-0.4, 0.5, 0.05) #9
        wspace = np.arange(-0.01, 0.02, 0.01)
        tmp = list(itertools.product(vspacex, vspacey))
        comb = [[v, vx, vy, 0] for vx, vy in tmp for v in [0,1]]
            
        #xy_combinations = list(itertools.product(vspacex, vspacey))
        #comb = [[v, x, y] for x, y in xy_combinations for v in range(len(gaits))]
       # breakpoint()
        
        def rearrange_pairs(sequence): # fa in modo che cambi o gait o vdes, wdes (dovrebbe essere piu facile)
            n = len(sequence)
            result = []

            # Iterate through the list two elements at a time
            for i in range(0, n, 4):
                # Add the first two elements in the current chunk
                if i + 1 < n:
                    result.append(sequence[i])
                    result.append(sequence[i + 1])
                
                # Add the next two elements but swap them
                if i + 3 < n:
                    result.append(sequence[i + 3])
                    result.append(sequence[i + 2])
            
            return result
        print("MODE: ", mode)
        comb = rearrange_pairs(comb)
        sim_time = 2#s
        simulator.run(
            simulation_time=sim_time,
            viewer=True,
            real_time=False,
            #visual_callback_fn=visual_callback,
            randomize=False,
            mode = mode,
            comb = [[1,0.6,0.,0.0],[1,0.2,0.2,0.0],[0,0,0,0],[0,0.3,0.3,0.02]]
        )
        
    ###
    elif mode==2: 
        print("pinocchio")
        robot.pin.info()
        print("mujoco")
        # RR_calf : 13
        # RL_calf : 10
        # FR_calf : 7
        # FL_calf : 4
        robot.mj.info()
        
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
        simulator = Simulator(robot.mj, controller) 
        visual_callback = (lambda viewer, step, q, v, data :
             desired_contact_locations_callback(viewer, step, q, v, data, controller))
        sim_time = 2#s reminder 1 sec = 1000 self.sim_step di simulator
        simulator.run(
            simulation_time=sim_time,
            use_viewer=False,
            real_time=False,
            #visual_callback_fn=visual_callback,
            randomize=0,
            mode = mode,
            comb = [[0,0.6,0.,0.02],[1,0,0,-0.03],[0,0,0,0],[0,0.3,0.3,0.02]],
            pert = True
                )
    ###
        
    elif mode == 999:
        5 
    
def TestPerb():
    # Load the MuJoCo model and data
    #model = mujoco.MjModel.from_xml_path('path_to_your_model.xml')  # Path to your MuJoCo model file

    cfg = Go2Config
    model = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    data = model.mj.data
    
    # Assuming the EE (End-Effector) indexes are [13, 10, 7, 4], corresponding to the feet in contact
    EE_indexes = [13, 10, 7, 4]

    # Initialize storage for the Jacobian matrix for all feet in contact
    nq = model.mj.nq  # Number of generalized coordinates (DOF)
    m = 3 * len(EE_indexes)  # Assuming 3D contact (x, y, z) per foot
    Jacobian = np.zeros((m, nq))  # Jacobian matrix

    # Loop through each foot in contact and compute its Jacobian
    for i, foot_id in enumerate(EE_indexes):
        # Temporary storage for the Jacobians (position and velocity)
        J_pos = np.zeros((3, nq))  # Jacobian for foot position
        J_rot = np.zeros((3, nq))  # Jacobian for foot orientation (optional if needed)

        # Compute the Jacobian for the foot
        mujoco.mj_jacBody(model, data, J_pos, J_rot, foot_id)

        # Store the position Jacobian in the overall Jacobian matrix
        Jacobian[i*3:(i+1)*3, :] = J_pos

    # Now apply the pseudoinverse and compute the perturbation as before

    # Define perturbations from a Gaussian distribution
    mu = 0
    sigma = 0.1
    delta_q = np.random.normal(mu, sigma, nq)  # Generalized coordinate perturbation
    delta_v = np.random.normal(mu, sigma, nq)  # Generalized velocity perturbation

    # Compute the pseudoinverse of the Jacobian
    Jacobian_pseudo_inv = np.linalg.pinv(Jacobian)

    # Project perturbations to contact-consistent space
    perturbation = np.dot((np.eye(nq) - np.dot(Jacobian_pseudo_inv, Jacobian)), np.concatenate((delta_q, delta_v)))

    # Separate the consistent perturbations for generalized coordinates and velocities
    delta_qc = perturbation[:nq]
    delta_vc = perturbation[nq:]

    # Apply the perturbation to the nominal state
    q_new = data.qpos + delta_qc  # Nominal generalized coordinates perturbed
    v_new = data.qvel + delta_vc  # Nominal generalized velocities perturbed
    
    # Update the simulation state with the new perturbed values
    # data.qpos = q_new
    # data.qvel = v_new
    # mujoco.mj_step(model, data)  # Step the simulation forward

    
    ###### Controller
    controller = BiConMPC(model.pin, replanning_time=0.05, sim_opt_lag=False)

    # controller.set_gait_params(trot)  # Choose between trot, jump and bound
    simulator = Simulator(model.mj, controller) 
    visual_callback = (lambda viewer, step, q, v, data :
            desired_contact_locations_callback(viewer, step, q, v, data, controller))
    sim_time = 2#s reminder 1 sec = 1000 self.sim_step di simulator
    simulator.run(
        simulation_time=sim_time,
        use_viewer=True,
        real_time=False,
        #visual_callback_fn=visual_callback,
        randomize=0,
        mode = 2,
        comb = [[0,0.6,0.,0.02]]
            )
    
if __name__ == "__main__":
    SimManager()
    #TestPerb()