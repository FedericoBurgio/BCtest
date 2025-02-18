import numpy as np
from typing import Any, Dict, Callable
from pathlib import Path
import itertools 

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

    def get_result_subfolder(policy: str, gait: int, vx: float, vy: float, sim_time: float, root_dir: str = "./results"):
        """
        Returns something like:
        ./results/policy_081059/gait1_vx0.10_vy-0.20
        Creates the directory if not existing.
        """
        # We build a short descriptive name for this combination
        combo_name = f"gait{gait}_vx{vx:.2f}_vy{vy:.2f}"

        # The subfolder might be results/mpc/... or results/081059/..., etc.
        if policy == "MPC":
            policy_subdir = "mpc"
        else:
            policy_subdir = policy  # e.g. "081059", "131519", "131330"

        # Build the full path
        subfolder_path = Path(root_dir) / policy_subdir / combo_name

        # Create directory if needed
        subfolder_path.mkdir(parents=True, exist_ok=True)

        return str(subfolder_path)

    
    def compute_rmse_wrt_commanded_velocities(self, file_path: str) -> dict:
        """
        Compute RMSE of actual (v_x, v_y) with respect to *piecewise* commanded v_x, v_y
        given that each segment is exactly sim_time * 1000 steps long.

        For example:
        comb = [
            [gait_index_0, vx_cmd_0, vy_cmd_0, w_z_cmd_0],
            [gait_index_1, vx_cmd_1, vy_cmd_1, w_z_cmd_1],
            ...
        ]
        and each segment is sim_time*1000 steps, so total steps = len(comb) * sim_time * 1000.

        :param file_path: Path to the HDF5 file containing the dataset 'v' (shape: [T, 6]).
        :return: Dictionary with RMSE values for (v_x, v_y), and combined v_xy if desired.
        """
        import numpy as np
        import h5py

        # Load the recorded velocities from the file
        with h5py.File(file_path, 'r') as f:
            velocities = f['v'][:]  # shape: (T, 6)
        
        # T is the total number of timesteps from the recorder
        T = velocities.shape[0]
        # N_segments is the number of commanded segments in comb
        N_segments = len(self.comb)

        # Compute how many steps we expect per segment
        # (based on your statement that each segment is exactly sim_time * 1000 steps)
        segment_steps = int(self.sim_time * 1000)

        # Prepare arrays for velocity differences
        diff_vx = np.zeros(T)
        diff_vy = np.zeros(T)

        # For each segment in comb, fill out the velocity difference
        for seg_idx in range(N_segments):
            vx_cmd = self.comb[seg_idx][1]  # commanded v_x
            vy_cmd = self.comb[seg_idx][2]  # commanded v_y

            start_idx = seg_idx * segment_steps
            end_idx = (seg_idx + 1) * segment_steps

            # In case T is not exactly multiple of segment_steps * N_segments,
            # we clamp end_idx to T.
            end_idx = min(end_idx, T)

            # Actual velocities in that segment
            actual_vx = velocities[start_idx:end_idx, 0]
            actual_vy = velocities[start_idx:end_idx, 1]

            # Compute difference for that segment
            diff_vx[start_idx:end_idx] = actual_vx - vx_cmd
            diff_vy[start_idx:end_idx] = actual_vy - vy_cmd

            # If end_idx == T, break early (avoid out-of-bounds for extra comb segments)
            if end_idx == T:
                break

        # Now compute MSE → RMSE across all timesteps
        mse_vx = np.mean(diff_vx ** 2)
        mse_vy = np.mean(diff_vy ** 2)

        rmse_vx = np.sqrt(mse_vx)
        rmse_vy = np.sqrt(mse_vy)
        rmse_total_vxy = np.sqrt((mse_vx + mse_vy) / 2.0)

        return {
            "rmse_vx": float(rmse_vx),
            "rmse_vy": float(rmse_vy),
            "rmse_total_vxy": float(rmse_total_vxy)
        }

    def compare_rmse_wrt_commanded_velocities(self):
        """
        1) Computes RMSE wrt commanded velocities for:
        - Nominal (MPC) file: self.savePathNom
        - Policy file: self.savePathPolicy

        2) Also computes z-velocity RMSE between the MPC and the policy.

        3) Saves all metrics to a JSON and CSV file.
        """
        import csv
        import json
        import h5py
        import numpy as np

        file_nominal = self.savePathNom   # HDF5 from MPC
        file_policy = self.savePathPolicy # HDF5 from policy

        # Output paths for JSON/CSV results
        save_json_path = file_policy.replace(".h5", "_cmd_vel_comparison.json")
        save_csv_path = file_policy.replace(".h5", "_cmd_vel_comparison.csv")

        # 1) Compute RMSE wrt commanded velocities for nominal (MPC) and policy
        nom_results = self.compute_rmse_wrt_commanded_velocities(file_nominal)
        pol_results = self.compute_rmse_wrt_commanded_velocities(file_policy)

        # 2) Compute z-velocity RMSE between nominal (MPC) vs. policy
        with h5py.File(file_nominal, 'r') as f_nom:
            vel_nom = f_nom['v'][:]  # shape (T1, 6)
        with h5py.File(file_policy, 'r') as f_pol:
            vel_pol = f_pol['v'][:]  # shape (T2, 6)

        # Ensure both runs have the same length for a fair comparison
        T = min(len(vel_nom), len(vel_pol))
        vel_nom = vel_nom[:T]
        vel_pol = vel_pol[:T]

        # z-velocity difference (index 2 if the order is [vx, vy, vz, wx, wy, wz])
        z_diff = vel_nom[:, 2] - vel_pol[:, 2]
        mse_z = np.mean(z_diff ** 2)
        rmse_z_mpc_vs_policy = float(np.sqrt(mse_z))

        # 3) Combine all results
        all_results = {
            "nominal_vs_commanded": nom_results,        # MPC vs. commanded
            "policy_vs_commanded": pol_results,         # Policy vs. commanded
            "z_vel_rmse_mpc_vs_policy": rmse_z_mpc_vs_policy
        }

        # ---- Save results to JSON ----
        with open(save_json_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        # ---- Save results to CSV ----
        with open(save_csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            # Header
            writer.writerow(["Comparison", "RMSE_vx", "RMSE_vy", "RMSE_total_vxy"])
            # MPC vs. commanded
            writer.writerow([
                "Nominal (MPC) vs Cmd",
                nom_results["rmse_vx"],
                nom_results["rmse_vy"],
                nom_results["rmse_total_vxy"]
            ])
            # Policy vs. commanded
            writer.writerow([
                "Policy vs Cmd",
                pol_results["rmse_vx"],
                pol_results["rmse_vy"],
                pol_results["rmse_total_vxy"]
            ])
            # Extra line for clarity
            writer.writerow([])
            # z-velocity RMSE (MPC vs. Policy)
            writer.writerow(["Z velocity RMSE (MPC vs. Policy)", rmse_z_mpc_vs_policy])

        print(f"Commanded velocity RMSE + Z-velocity comparison saved to:\n"
            f"  JSON: {save_json_path}\n"
            f"  CSV:  {save_csv_path}")
        return all_results


    def mpc_rmse_wrt_commanded_velocities(self):
        """
RMSE of the *MPC (nominal)* data
        with respect to commanded velocities (self.comb).
        Call this *immediately after* self.runMPC().
        """
        import csv
        import json

        file_mpc = self.savePathNom  # <- nominal .h5 that runMPC() just created
        results = self.compute_rmse_wrt_commanded_velocities(file_mpc)

        # Create new file paths to avoid overwriting policy results
        save_json_path = file_mpc.replace(".h5", "_mpc_cmd_vel_comparison.json")
        save_csv_path = file_mpc.replace(".h5", "_mpc_cmd_vel_comparison.csv")

        # Package results
        all_results = {
            "mpc_vs_commanded": results
        }

        # Save to JSON
        with open(save_json_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        # Save to CSV
        with open(save_csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Comparison", "RMSE_vx", "RMSE_vy", "RMSE_total_vxy"])
            writer.writerow([
                "MPC vs Cmd",
                results["rmse_vx"],
                results["rmse_vy"],
                results["rmse_total_vxy"]
            ])

        print(f"**MPC** commanded velocity RMSE results saved to:\n"
            f"  {save_json_path}\n"
            f"  {save_csv_path}\n")

        return results


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
            PerfRec.mpc_rmse_wrt_commanded_velocities()
            for i in ["081059","131519","131330"]:
                print(i)
                PerfRec = PerformanceRecorder([comb_i], sim_time, i, ep)
                PerfRec.rep(i)
                PerfRec.compare_rmse_wrt_commanded_velocities()
        else:    
            PerfRec = PerformanceRecorder([comb_i], sim_time, policy, ep)
            PerfRec.rep(policy)
            PerfRec.runMPC()
            PerfRec.mpc_rmse_wrt_commanded_velocities()
else: 
    if policy == "all":
        PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
        PerfRec.runMPC()
        PerfRec.mpc_rmse_wrt_commanded_velocities()
            
        for i in ["081059","131519","131330"]:
            print(i)
            PerfRec = PerformanceRecorder(comb, sim_time, i, ep)
            PerfRec.rep(i)
            PerfRec.compare_rmse_wrt_commanded_velocities()
    else:    
        PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
        PerfRec.rep(policy)
        PerfRec.runMPC()
        PerfRec.mpc_rmse_wrt_commanded_velocities()