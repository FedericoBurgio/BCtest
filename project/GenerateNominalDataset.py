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

from Sampler import sobol

def generate_or_load_sobol_samples(filename, n_samples, l_bounds, u_bounds, seed):
    if os.path.exists(filename):
        print(f"Loading Sobol samples from {filename}")
        comb = np.load(filename)['samples']
    else:
        print(f"Generating Sobol samples and saving to {filename}")
        comb = sobol(n_samples, l_bounds, u_bounds, seed)
        np.savez(filename, samples=comb)
    return comb

def recNom(comb, datasetName):
    ##Get nominal 
    gaits = [trot, jump]
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
    recorder = Recorders.DataRecorderNominal(controller)
    simulator = Simulator(robot, controller, recorder)
    recorder.gait_index = int(comb[0][0])
    
    sim_time_nom = gaits[int(comb[0][0])].gait_period
    simulator.run( ##NOMINAL
        simulation_time=sim_time_nom,#.5 = 1 cycle
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb,
        verbose = False
    )
    if not simulator.controller.diverged:
        recorder.save_data_hdf5(datasetName) #dataset e .h5 gia nella funzione
        return True
    else:
        return False

def RecordNominalDataset(datasetName, comb_ = []):        
    Jtemp = "temp/progressJ_" + datasetName + ".npz"
    try:
        data = np.load(Jtemp)
        j = data['j']        
    except FileNotFoundError:
        j= 0  
    if len(comb_) == 0:
        
        n_samples = 256
        l_bounds =  [0, -0.1, -0.3, -0.07] 
        u_bounds =  [1, 0.5, 0.3, 0.07] 
        seed = 963
        sobol_file = "datasets/sobol_samples_" + str(n_samples) + str(l_bounds) +str(u_bounds) + str(seed) + ".npz"
        
        datasetName = str(n_samples) + str(l_bounds) +str(u_bounds) + str(seed) + datasetName
        comb = generate_or_load_sobol_samples(sobol_file, n_samples, l_bounds, u_bounds, seed)
    else:
        comb = comb_
    jj = 0
    if j < len(comb):
        while jj < 5:
            print("j: ", j, "/",len(comb), "\ncomb[j]: ",comb[j])
            recNom([comb[j]], datasetName)
            #save Jtemp
            j = j + 1
            np.savez(Jtemp, j=j)
            jj = jj + 1 
        
# Record_with_pert(7,             #num samples per cycle - 
#                 .7,            #pertubed episode duration
#                 "yo",#"testGrid10_1-2BIScon05",
#                 comb_= [[1,.3,.3,0]])       #saved dataset name
import itertools
import numpy as np
vx = np.arange(-0.1,0.6,0.1)
vy = np.arange(-0.3,0.3,0.1)
vy = np.arange(-0.3,0.4,0.1)
w = np.arange(-0.07,0.14,0.07)
comb = list(itertools.product([0,1],vx,vy,w))


RecordNominalDataset(               #num samples per cycle - 
                                 #pertubed episode duration
                "girdNominal",     #"testGrid10_1-2BIScon05",
                comb_ = comb)        #saved dataset name