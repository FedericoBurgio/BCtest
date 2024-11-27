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

import h5py

gaits = [trot, jump]
MAXkk = 6 #max no of iterations per cycle - needed to prevent crashes 

from applyPert import apply_perturbation
 
def rec(comb, q0, v0, cntBools, sim_time, datasetName):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    q0, v0 = apply_perturbation(q0, v0, cntBools, robot, gait=comb[0][0])
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    recorder = Recorders.DataRecorderPD(controller)
    simulator = Simulator(robot, controller, recorder)
    #robot.reset(q0, v0)
    #FR foot 26 RL foot 42 RR foot 54
    #fl 13 fr 25 rl 41 rr 53 "foot geometries"
    #sim_time = 1.8  # seconds
    recorder.gait_index = comb[0][0]
    simulator.run(
        simulation_time=sim_time,
        use_viewer = 1,
        real_time = False,
        visual_callback_fn=None,
        randomize=True,
        comb = comb,
        verbose = False
    )
    
    if not simulator.controller.diverged:
        simulator.data_recorder.save_data_hdf5(datasetName)
        return True
    else:
        return False

def generate_uniform_array(n, range_step, seed=None):
    """
    Generates an array of random values, each sampled from a unique uniform range.
    
    Parameters:
    - n (int): Number of random values to generate.
    - range_step (int): The size of the range for each random value (e.g., 50).
    - seed (int, optional): Seed for reproducibility. Defaults to None.
    
    Returns:
    - np.ndarray: Array of random indices.
    """
    rng = np.random.default_rng(seed)
    random_indices = np.array([rng.uniform(i * range_step, (i + 1) * range_step) for i in range(n)])
    return np.round(random_indices).astype(int)  # Ensure indices are integers

def loadNominal(filename, n_samples=100, range_step=100, seed=None):
    
    
    """
    Load or sample the nominal trajectory data from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file containing nominal trajectory data.
        sampled_filename (str): Path to the HDF5 file to store or load sampled data.
        n_samples (int): Number of samples to extract.
        range_step (int): Step size for generating uniform sample ranges.
        seed (int, optional): Seed for reproducibility.

    Returns:
        dict: Dictionary containing sampled states, velocities, and contact booleans.
    """
    sampled_filename = "datasets/sampled_" + filename + ".h5"
    filename = "datasets/" + filename + ".h5"
    # Check if the sampled dataset already exists
    
    if os.path.exists(sampled_filename):
        print(f"Loading pre-sampled data from {sampled_filename}")
        try:
            with h5py.File(sampled_filename, 'r') as sf:
                states = sf['states'][:]
                velocities = sf['velocities'][:]
                contact_bools = sf['contact_bools'][:]
                gait_des = sf['gait_des'][:]
            return states, velocities, contact_bools, gait_des
        except Exception as e:
            print(f"Error loading sampled file: {e}")
            return None

    # Otherwise, sample from the original dataset
    print(f"Sampling data from {filename}")
    try:
        with h5py.File(filename, 'r') as f:
            data_length = len(f['states'])
            #n_samples = len((f['states'][:])[::50])
            n_samples = len(f['states'][:])//range_step
            # indices = generate_uniform_array(n_samples, range_step, seed)
            # indices = np.clip(indices, 0, data_length - 1)  # Ensure indices are within bounds

            # Sample data
            states = f['states'][::500]
            velocities = f['v'][::500]
            contact_bools = f['cnt_bools'][::500]
            gait_des = f['gait_des'][::500]

        # Save the sampled data for future use
        with h5py.File(sampled_filename, 'w') as sf:
            sf.create_dataset('states', data=states)
            sf.create_dataset('velocities', data=velocities)
            sf.create_dataset('contact_bools', data=contact_bools)
            sf.create_dataset('gait_des', data=gait_des)

        print(f"Sampled data saved to {sampled_filename}")
        return states, velocities, contact_bools, gait_des

    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except Exception as e:
        print(f"Error while sampling data: {e}")
        return None

def recPert(sim_time, nominalDatasetName, datasetName):
    ##Record with perturbation
    
    qNom, vNom, cntBools, gait_des = loadNominal(nominalDatasetName)
 
    Itemp = "temp/progressI_" + datasetName + ".npz"
    
    try:
        data = np.load(Itemp)
        i = data['i']
    except FileNotFoundError:
        i = 0  
    kk = 0
    while i < len(qNom): 
        if kk > MAXkk : break
        print(i,"/",len(cntBools))
        print(cntBools[i])
        print(gait_des[i])
        if rec([gait_des[i]], qNom[i], vNom[i], cntBools[i], sim_time, datasetName):    
            i += 1
            np.savez(Itemp, i=i)
        print(i, "/", len(qNom))
        kk+=1

 
recPert(4,"girdNominal", "ForcesGird_4ec")   