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

import h5py
import Controllers
import Recorders

gaits = [trot, jump]
class PerformanceRecorder():
    def __init__(self, comb, sim_time, policy, ep):
        self.comb = comb
        self.sim_time = sim_time
        self.policy = policy
        self.ep = ep
        self.savePathPolicy = "/home/atari_ws/project/results/tracking_perf_" + policy + str(comb) + str(sim_time) + ".h5"
        self.savePathNom = "/home/atari_ws/project/results/tracking_perf_Nom" + str(comb) + str(sim_time) + ".h5"
        
    def runMPC(self):
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
                *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                rotor_inertia=cfg.rotor_inertia,
                gear_ratio=cfg.gear_ratio,
                )
        controller = BiConMPC(robot.pin, replanning_time=0.05, sim_opt_lag=False)
        recorder = Recorders.DataRecorderNominal(controller)
        simulator = Simulator(robot, controller, recorder)
        recorder.gait_index = self.comb[0][0]

        simulator.run( ##NOMINAL
            simulation_time= self.sim_time,#.5 = 1 cycle
            use_viewer= VIEW,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb = self.comb,
            verbose = False
        )

        recorder.save_data_hdf5_tracking_perf(self.savePathNom)

    def rep(self, policy = None):
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
                *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
                rotor_inertia=cfg.rotor_inertia,
                gear_ratio=cfg.gear_ratio,
                )
        path = "/home/atari_ws/project/models/" + self.policy + "/best_policy_ep" + self.ep + ".pth"
        controller = Controllers.TrainedControllerPD(robot.pin, datasetPath = path)
        if policy != None: controller. policyID = policy
        if controller.ID == "SEQ":
            self.savePathPolicy = self.savePathPolicy.replace(".h5", "_SEQ.h5")

        recorder = Recorders.DataRecorderNominal(controller)
        simulator = Simulator(robot, controller, recorder)
        controller.gait_index = self.comb[0][0]

        simulator.run(
            simulation_time = self.sim_time,
            use_viewer= VIEW,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb = self.comb,
            verbose = False
        )

        recorder.save_data_hdf5_tracking_perf(self.savePathPolicy)
    
    def run(self):
        self.rep()
        self.runMPC()
    
    def run_all(self):
        print("MPC")
        self.runMPC()
        
        print("cont + vel, no index")
        self.policy = "081059"
        self.savePathPolicy = "/home/atari_ws/project/results/tracking_perf_" + policy + str(comb) + str(sim_time) + ".h5"
        self.savePathNom = "/home/atari_ws/project/results/tracking_perf_Nom" + str(comb) + str(sim_time) + ".h5"
        self.rep(self.policy)
        self.compare_trajectories()
        
        print("vel")
        self.policy = "131519"
        self.savePathPolicy = "/home/atari_ws/project/results/tracking_perf_" + policy + str(comb) + str(sim_time) + ".h5"
        self.savePathNom = "/home/atari_ws/project/results/tracking_perf_Nom" + str(comb) + str(sim_time) + ".h5"
        self.rep(self.policy)
        self.compare_trajectories()
        
        print("cont")
        self.policy = "131330"
        self.savePathPolicy = "/home/atari_ws/project/results/tracking_perf_" + policy + str(comb) + str(sim_time) + ".h5"
        self.savePathNom = "/home/atari_ws/project/results/tracking_perf_Nom" + str(comb) + str(sim_time) + ".h5"
        self.rep(self.policy)
        self.compare_trajectories()
    
    def compare_trajectories(self):
        import csv
        import json
        file_nominal = self.savePathNom
        file_policy = self.savePathPolicy   
        save_json_path = file_policy.replace(".h5", "_comparison_results.json")
        save_csv_path = file_policy.replace(".h5", "_comparison_results.csv")
        
        with h5py.File(file_nominal, 'r') as f_nom, h5py.File(file_policy, 'r') as f_policy:
            # Load datasets
            states_nom = f_nom['states'][:]
            states_policy = f_policy['states'][:]
            velocities_nom = f_nom['v'][:]
            velocities_policy = f_policy['v'][:]
            
            # Ensure all datasets have the same length
            min_length = min(len(states_nom), len(states_policy), len(velocities_nom), len(velocities_policy))
            states_nom = states_nom[:min_length]
            states_policy = states_policy[:min_length]
            velocities_nom = velocities_nom[:min_length]
            velocities_policy = velocities_policy[:min_length]

            # Compute RMSE for positions (states)
            mse_states = np.mean((states_nom - states_policy) ** 2, axis=0)
            rmse_states = np.sqrt(mse_states)
            total_rmse_states = np.sqrt(np.mean(mse_states))
            
            # Compute RMSE for velocities (v)
            mse_velocities = np.mean((velocities_nom - velocities_policy) ** 2, axis=0)
            rmse_velocities = np.sqrt(mse_velocities)
            total_rmse_velocities = np.sqrt(np.mean(mse_velocities))

            # CoM analysis (as discussed above)
            # Extract CoM positions
            com_nom = states_nom[:, :3]
            com_policy = states_policy[:, :3]

            # CoM trajectory deviation
            com_diff = np.linalg.norm(com_nom - com_policy, axis=1)
            mae_com = np.mean(com_diff)
            mse_com = np.mean(com_diff ** 2)
            rmse_com = np.sqrt(mse_com)

            # Vertical stability
            com_z_nom = states_nom[:, 2]
            com_z_policy = states_policy[:, 2]
            std_z_nom = np.std(com_z_nom)
            std_z_policy = np.std(com_z_policy)
            mse_z = np.mean((com_z_nom - com_z_policy) ** 2)
            rmse_z = np.sqrt(mse_z)

            # Update results dictionary
            results = {
                "rmse_states_per_variable": rmse_states.tolist(),
                "total_rmse_states": total_rmse_states,
                "rmse_velocities_per_variable": rmse_velocities.tolist(),
                "total_rmse_velocities": total_rmse_velocities,
                "mae_com": mae_com,
                "rmse_com": rmse_com,
                "std_z_nom": std_z_nom,
                "std_z_policy": std_z_policy,
                "rmse_z": rmse_z,
                # Add other metrics as needed
            }

        # Save results to a JSON file
        with open(save_json_path, "w") as file:
            json.dump(results, file, indent=4)

        # Save results to a CSV file
        with open(save_csv_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Values"])
            writer.writerow(["RMSE States Per Variable", rmse_states.tolist()])
            writer.writerow(["Total RMSE States", total_rmse_states])
            writer.writerow(["RMSE Velocities Per Variable", rmse_velocities.tolist()])
            writer.writerow(["Total RMSE Velocities", total_rmse_velocities])

        print(f"Results saved to {save_json_path} and {save_csv_path}")
        return rmse_states, total_rmse_states, rmse_velocities, total_rmse_velocities
    
from compareConf import policy, ep, comb, sim_time, multi, view
VIEW = view
if multi:
    for comb_i in comb:
        if policy == "all":
            PerfRec = PerformanceRecorder([comb_i], sim_time, policy, ep)
            PerfRec.runMPC()
            for i in ["081059","131519","131330"]:
                print(i)
                PerfRec = PerformanceRecorder([comb_i], sim_time, i, ep)
                PerfRec.rep(i)
                PerfRec.compare_trajectories()
        else:    
            PerfRec = PerformanceRecorder([comb_i], sim_time, policy, ep)
            PerfRec.rep(policy)
            PerfRec.runMPC()
            PerfRec.compare_trajectories()
else: 
    if policy == "all":
        PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
        PerfRec.runMPC()
        for i in ["081059","131519","131330"]:
            print(i)
            PerfRec = PerformanceRecorder(comb, sim_time, i, ep)
            PerfRec.rep(i)
            PerfRec.compare_trajectories()
    else:    
        PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
        PerfRec.rep(policy)
        PerfRec.runMPC()
        PerfRec.compare_trajectories()