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
    recorder.gait_index = comb[0][0]
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
    recorder.gait_index = comb[0][0]
    sim_time_nom = 0.5
    simulator.run( ##NOMINAL
        simulation_time=sim_time_nom,#.5 = 1 cycle
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )
    
    Step_ = (int(1000*sim_time_nom)-1)//(noStep-1)
    qNom, vNom, cntBools = recorder.getNominal(Step_)#get in each Step_ ([::Step_])
    
    del cfg
    del robot
    del controller
    del recorder
    del simulator

    i = 0
    kk=0
    Itemp = "progressI_" + datasetName + ".npz"
    
    try:
        data = np.load(Itemp)
        i = data['i']
    except FileNotFoundError:
        i = 0  

    while i < len(qNom): 
        if kk > 6 : break
        print(i,"/",len(cntBools))
        print(cntBools[i])
        if rec(comb, qNom[i], vNom[i], cntBools[i], sim_time, datasetName):    
            i += 1
            np.savez(Itemp, i=i)
        print(i, "/", len(qNom))
        kk+=1
        
    if i == len(qNom):
        i = 0
        np.savez(Itemp, i=i)
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
 
def Record_with_pert(noStep, sim_time, datasetName, numSamples = 1, comb_ = []):
        
    sampling = "grid"
    
    if len(comb_) == 0:
        if sampling == "lhs":
            from Sampler import LHS
            comb = LHS(gaitsI, vspacex, vspacey, wspace, numSamples, seed_=77)
        elif sampling == "grid": 
            import itertools
            
            gaitsI = np.array([0, 1])
            vspacex = np.arange(-0.1, 0.6, 0.08)
            vspacey = np.arange(-0.4, 0.5, 0.1)
            wspace = np.arange(-0.07, 0.14, 0.07)
            comb0 = list(itertools.product([0], vspacex, [0], [0]))
            np.random.seed(77)  # For reproducibility
            indices = np.random.choice(len(comb0), size=100, replace=False)
            comb = np.array(comb0)[indices]
            np.random.seed(None)
            del comb0
        elif sampling == "sobol":
            from Sampler import sobol
            comb = sobol()
            
    else:
        comb = comb_
    Jtemp = "progressJ_" + datasetName + ".npz"

    try:
        data = np.load(Jtemp)
        j = data['j']        
    except FileNotFoundError:
        j= 0  
    
    if j < len(comb):
        print("j: ", j, "/",len(comb), "\ncomb[j]: ",comb[j])
        if recPert([comb[j]], noStep, sim_time, datasetName):
            j = j + 1 
            np.savez(Jtemp, j=j)
        else:
            np.savez(Jtemp, j=j)
    
if __name__ == "__main__":
    #replay("090811", "60", comb = [[0,0.2,0,0],[1,0.4,0,0],[0,0,0,0],[0,0.3,0.3,0.05]], sim_time=3)
    
    #replay("101525", "150", comb = [[0,0.3,0,0]], sim_time=3)
    z9 = np.zeros(9)
    grid = np.vstack([z9,np.arange(-0.1,0.6,0.08),z9,z9]).T
    Record_with_pert(10,             #num samples per cycle - 
                    0.5,            #pertubed episode duration
                    "testGrid10_1-2BIScon05",
                    comb_= grid)       #saved dataset name
    