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
    controller.policyID = dataset
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
        

#from compareConf import policy, ep, comb, sim_time 
#policy = policy

policy = "081059" #081059: ibrido. 131519: vel. 131330: contact
ep = "final"
multi = False #se True valuta le commanded una ad una - SETTARE TRUE PER FARE RECORDING PER HEATMAPS
view = True
comb = [[0,.3,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]
sim_time = 3
 
replay(policy, ep, comb, sim_time)