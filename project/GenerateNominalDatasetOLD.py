import copy
import numpy as np
import os
from typing import Any, Dict, Callable
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
import Sampler

import torch
import torch.nn as nn

import nets
import Controllers
import Recorders
import time

# Define available gaits
gaits = [trot, jump]

from scipy.stats import qmc

# Global configuration
sobol_sample_file = "sobol_samples"  # File to store the Sobol samples
progress_file = "progress_sobol.npz"  # File to track progress


# Sobol sampling function (called only once)
def generate_sobol_samples(filename, n_samples=30, seed=220):
    n_dimensions = 4
    l_bounds = [0, 0, -0.3, -0.07]  # Lower bounds
    u_bounds = [1, 0.5, 0.3, 0.07]  # Upper bounds

    # Create a Sobol sampler
    sampler = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale samples
    scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

    # Round gait index to discrete values
    scaled_samples[:, 0] = np.round(scaled_samples[:, 0])

    # Save the samples for reuse
    np.save(filename, scaled_samples)
    print(f"Sobol samples saved to {filename}")

# Recording function
def rec(comb, dataset_name):
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
    sim_time_nom = 0.5  # Simulation time for nominal trajectory
    simulator.run(
        simulation_time=sim_time_nom,
        use_viewer=0,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb=comb,
        verbose=False,
    )
    if not simulator.controller.diverged:
        simulator.data_recorder.save_data_hdf5(dataset_name)
        return True
    else:
        return False

# Load and process Sobol samples with progress tracking
def load_and_record_with_progress(sample_file, progress_file, dataset_name):
    
    # Load Sobol samples
    comb = np.load(sample_file)

    # Load progress or start fresh
    try:
        progress_data = np.load(progress_file)
        j = progress_data['j']
    except FileNotFoundError:
        j = 0

    # Process remaining combinations
    while j < len(comb):
        print(f"Processing j: {j}/{len(comb)} | comb[j]: {comb[j]}")

        # Run the recording process for the current combination
        success = rec([comb[j].tolist()], f"{dataset_name}_{j}.h5") 

        if success:
            j += 1  # Increment progress only on success
        np.savez(progress_file, j=j)  # Save progress

n_samples = 100  # Number of Sobol samples
dataset_name = "robot_nominal_data"

# Generate Sobol samples only if the file does not exist
if not os.path.exists(sobol_sample_file):
    generate_sobol_samples(sobol_sample_file, n_samples=n_samples)

# Load Sobol samples and start recording with progress tracking
load_and_record_with_progress(sobol_sample_file, progress_file, dataset_name)
