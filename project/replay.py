import numpy as np
from typing import Any, Dict, Callable

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

import Controllers

def replay(dataset, ep, comb, sim_time):
    cfg = Go2Config
    robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
            )
    path = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth"
    # NNprop = np.load("/home/atari_ws/project/models/" + dataset + "/NNprop.npz")
    controller = Controllers.TrainedControllerPD(robot.pin, datasetPath = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth")#, state_size=NNprop['s_size'], action_size=NNprop['a_size'], hL=NNprop['h_layers'],
                                                    #replanning_time=0.05, sim_opt_lag=False,
                                                    #datasetPath = "/home/atari_ws/project/models/" + dataset + "/best_policy_ep" + ep + ".pth")
    controller.gait_index = comb[0][0]
    simulator = Simulator(robot, controller)
    
    simulator.run(
        simulation_time = sim_time,
        viewer=True,
        real_time=False,
        visual_callback_fn=None,
        randomize=False,
        comb = comb
    )
        
# datasetname = "151143" #full state
# datasetname= "151216"  #senza states = np.delete(states, [73,74,75,76,77], axis=1)
# datasetname = "151306" #BUONO3

# datasetname = "160949" #BUONO3

# datasetname = "161519" 
# datasetname = "161442" 

# datasetname = "161523"
# datasetname = "161539"

# datasetname = "161539"
# datasetname = "171411"

# datasetname = "171520" #50 - 0.5
# datasetname = "171557" #50 - 0.75
# datasetname = "180902"

# datasetname = "190904"

datasetname = "191509"
datasetname = "181955"

datasetname = "230946"
datasetname = "241544"

datasetname = "241748"

datasetname = "251153"
datasetname = "241755"

datasetname = "261004"
datasetname = "260954"
datasetname = "261122"
datasetname = "261243"
datasetname = "261643"
datasetname = "261754"
ep = "final"
replay(datasetname, ep, [[0, 0.3, 0.2, 0.0],[0, 0, 0.0, 0.0],[1, 0.3, 0.0, 0.0],[0, -0.3, 0.0, 0.0],[1,0.2,0.2,0]], 1.2)