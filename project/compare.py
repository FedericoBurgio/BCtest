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
import csv
import json
import os

gaits = [trot, jump]

class PerformanceRecorder():
    def __init__(self, comb, sim_time, policy, ep):
        self.comb = comb
        self.sim_time = sim_time
        self.policy = policy
        self.ep = ep
        self.savePathPolicy = f"/home/atari_ws/project/results/tracking_perf_{policy}{comb}{sim_time}.h5"
        self.savePathNom = f"/home/atari_ws/project/results/tracking_perf_Nom{comb}{sim_time}.h5"

    def runMPC(self):
        """
        Always runs the MPC simulation, stores .h5
        """
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

        simulator.run(
            simulation_time=self.sim_time,
            use_viewer=VIEW,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb=self.comb,
            verbose=False
        )

        recorder.save_data_hdf5_tracking_perf(self.savePathNom)

    def runPolicy(self, policy=None):
        """
        Always runs the Policy simulation, stores .h5
        """
        cfg = Go2Config
        robot = MJPinQuadRobotWrapper(
            *RobotModelLoader.get_paths(cfg.name, mesh_dir=cfg.mesh_dir),
            rotor_inertia=cfg.rotor_inertia,
            gear_ratio=cfg.gear_ratio,
        )
        path = f"/home/atari_ws/project/models/{self.policy}/best_policy_ep{self.ep}.pth"
        controller = Controllers.TrainedControllerPD(robot.pin, datasetPath=path)
        if policy:
            controller.policyID = policy
        if controller.ID == "SEQ":
            self.savePathPolicy = self.savePathPolicy.replace(".h5", "_SEQ.h5")

        recorder = Recorders.DataRecorderNominal(controller)
        simulator = Simulator(robot, controller, recorder)
        controller.gait_index = self.comb[0][0]

        simulator.run(
            simulation_time=self.sim_time,
            use_viewer=VIEW,
            real_time=False,
            visual_callback_fn=None,
            randomize=False,
            comb=self.comb,
            verbose=False
        )

        recorder.save_data_hdf5_tracking_perf(self.savePathPolicy)

    # ---------------------------------------------------------------------------
    # "Maybe" run sim if .h5 doesn't exist, otherwise skip
    # ---------------------------------------------------------------------------
    def maybe_runMPC(self):
        if os.path.isfile(self.savePathNom):
            print(f"[SKIP runMPC] Found existing {self.savePathNom}")
        else:
            print(f"[RUN runMPC] .h5 not found, simulating MPC → {self.savePathNom}")
            self.runMPC()

    def maybe_runPolicy(self, policy=None):
        if os.path.isfile(self.savePathPolicy):
            print(f"[SKIP runPolicy] Found existing {self.savePathPolicy}")
        else:
            print(f"[RUN runPolicy] .h5 not found, simulating policy → {self.savePathPolicy}")
            self.runPolicy(policy)

    # ---------------------------------------------------------------------------
    # Compute metrics from existing .h5 data without re-running
    # ---------------------------------------------------------------------------
    def compute_metrics_from_existing(self):
        """
        If both self.savePathNom and self.savePathPolicy exist,
        compute metrics (RMSE, MAE, Z-Error) from the .h5 files directly,
        skipping any simulation.
        """
        if not os.path.isfile(self.savePathNom):
            print(f"[ERROR] {self.savePathNom} does not exist, cannot compute metrics.")
            return
        if not os.path.isfile(self.savePathPolicy):
            print(f"[ERROR] {self.savePathPolicy} does not exist, cannot compute metrics.")
            return

        # Compute metrics for MPC
        self.compute_mpc_metrics()

        # Compare policy vs commanded velocities + Z-velocity
        self.compare_metrics()

    # ---------------------------------------------------------------------------
    # Metrics computations
    # ---------------------------------------------------------------------------
    def compute_metrics(self, file_path: str) -> dict:
        """
        Computes RMSE, MAE, and cumulative MAE from the .h5 file.

        Parameters:
            file_path (str): Path to the .h5 file.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        with h5py.File(file_path, 'r') as f:
            velocities = f['v'][:]  # shape: (T, 6)

        T = velocities.shape[0]
        N_segments = len(self.comb)
        # Assuming 1 kHz simulation -> each segment = sim_time * 1000 steps
        segment_steps = int(self.sim_time * 1000)

        diff_vx = np.zeros(T)
        diff_vy = np.zeros(T)
        diff_z = np.zeros(T)

        for seg_idx in range(N_segments):
            vx_cmd = self.comb[seg_idx][1]
            vy_cmd = self.comb[seg_idx][2]
            # Assuming commanded z velocity is zero. Modify if needed.
            #vz_cmd = 0.0

            start_idx = seg_idx * segment_steps
            end_idx = (seg_idx + 1) * segment_steps
            end_idx = min(end_idx, T)

            actual_vx = velocities[start_idx:end_idx, 0]
            actual_vy = velocities[start_idx:end_idx, 1]
            #actual_vz = velocities[start_idx:end_idx, 2]

            diff_vx[start_idx:end_idx] = actual_vx - vx_cmd
            diff_vy[start_idx:end_idx] = actual_vy - vy_cmd
            #diff_z[start_idx:end_idx] = actual_vz - vz_cmd

            if end_idx == T:
                break

        # --- RMSE ---
        mse_vx = np.mean(diff_vx ** 2) if T > 0 else 0.0
        mse_vy = np.mean(diff_vy ** 2) if T > 0 else 0.0
        rmse_vx = float(np.sqrt(mse_vx))
        rmse_vy = float(np.sqrt(mse_vy))
        rmse_total_vxy = float(np.sqrt(mse_vx + mse_vy))

        # --- MAE (Mean Absolute Error) ---
        abs_vx = np.abs(diff_vx)
        abs_vy = np.abs(diff_vy)
        #abs_vz = np.abs(diff_z)
        mae_vx = float(np.mean(abs_vx)) if T > 0 else 0.0
        mae_vy = float(np.mean(abs_vy)) if T > 0 else 0.0
        #mae_vz = float(np.mean(abs_vz)) if T > 0 else 0.0

        # If you want the mean of the 2D velocity error magnitude (vector norm):
        mae_total_vxy = float(np.mean(np.sqrt(diff_vx**2 + diff_vy**2))) if T > 0 else 0.0

        # --- Cumulative MAE (sum of absolute errors) ---
        # (If you want time-integrated error, multiply by dt as well.)
        cumulative_mae_vx = float(np.sum(abs_vx))
        cumulative_mae_vy = float(np.sum(abs_vy))
        #cumulative_mae_vz = float(np.sum(abs_vz))
        cumulative_mae_total_vxy = float(np.sum(np.sqrt(diff_vx**2 + diff_vy**2)))

        return {
            "rmse_vx": rmse_vx,
            "rmse_vy": rmse_vy,
            "rmse_total_vxy": rmse_total_vxy,

            "mae_vx": mae_vx,
            "mae_vy": mae_vy,
            #"mae_vz": mae_vz,
            "mae_total_vxy": mae_total_vxy,

            "cumulative_mae_vx": cumulative_mae_vx,
            "cumulative_mae_vy": cumulative_mae_vy,
            #"cumulative_mae_vz": cumulative_mae_vz,
            "cumulative_mae_total_vxy": cumulative_mae_total_vxy,

            "survived_timesteps": T
        }

    def compute_mpc_metrics(self):
        """
        Computes and saves the metrics for the MPC (nominal) simulation.
        """
        results = self.compute_metrics(self.savePathNom)

        save_json_path = self.savePathNom.replace(".h5", "_cmd_vel_comparison.json")
        save_csv_path = self.savePathNom.replace(".h5", "_cmd_vel_comparison.csv")

        all_results = {
            "mpc_vs_commanded": results
        }

        # Save to JSON
        with open(save_json_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        # Save to CSV
        with open(save_csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "Comparison",
                "RMSE_vx", "RMSE_vy", "RMSE_total_vxy",
                "MAE_vx", "MAE_vy", #"MAE_vz", 
                "MAE_total_vxy",
                "Cumulative_MAE_vx", "Cumulative_MAE_vy", #"Cumulative_MAE_vz", 
                "Cumulative_MAE_total_vxy",
                "Survived_Timesteps"
            ])
            writer.writerow([
                "MPC vs Cmd",
                results["rmse_vx"],
                results["rmse_vy"],
                results["rmse_total_vxy"],
                results["mae_vx"],
                results["mae_vy"],
                #results["mae_vz"],
                results["mae_total_vxy"],
                results["cumulative_mae_vx"],
                results["cumulative_mae_vy"],
                #results["cumulative_mae_vz"],
                results["cumulative_mae_total_vxy"],
                results["survived_timesteps"]
            ])

        print(f"**MPC** commanded vel metrics saved → {save_json_path}, {save_csv_path}")
        return all_results

    def compare_metrics(self):
        """
        Compares policy metrics against nominal (MPC) metrics and computes
        Z-velocity RMSE and MAE.
        """
        file_nominal = self.savePathNom
        file_policy = self.savePathPolicy

        save_json_path = file_policy.replace(".h5", "_cmd_vel_comparison.json")
        save_csv_path = file_policy.replace(".h5", "_cmd_vel_comparison.csv")

        # Compute metrics
        nom_results = self.compute_metrics(file_nominal)
        pol_results = self.compute_metrics(file_policy)

        # Compute Z-velocity RMSE and MAE (policy vs MPC).
        with h5py.File(file_nominal, 'r') as f_nom, h5py.File(file_policy, 'r') as f_pol:
            vel_nom = f_nom['v'][:, 2]  # Z-velocity (nominal)
            vel_pol = f_pol['v'][:, 2]  # Z-velocity (policy)

        T = min(len(vel_nom), len(vel_pol))
        if T > 0:
            z_diff = vel_nom[:T] - vel_pol[:T]
            mse_z = np.mean(z_diff ** 2)
            rmse_z = float(np.sqrt(mse_z))
            mae_z = float(np.mean(np.abs(z_diff)))
        else:
            rmse_z = np.nan
            mae_z = np.nan

        all_results = {
            "nominal_vs_commanded": nom_results,
            "policy_vs_commanded": pol_results,
            "z_vel_rmse_mpc_vs_policy": rmse_z,
            "z_vel_mae_mpc_vs_policy": mae_z
        }

        # Save to JSON
        with open(save_json_path, "w") as json_file:
            json.dump(all_results, json_file, indent=4)

        # Save to CSV
        with open(save_csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "Comparison",
                "RMSE_vx", "RMSE_vy", "RMSE_total_vxy",
                "MAE_vx", "MAE_vy", #"MAE_vz", 
                "MAE_total_vxy",
                "Cumulative_MAE_vx", "Cumulative_MAE_vy", #"Cumulative_MAE_vz", 
                "Cumulative_MAE_total_vxy",
                "Survived_Timesteps"
            ])

            # Nominal
            writer.writerow([
                "Nominal (MPC) vs Cmd",
                nom_results["rmse_vx"],
                nom_results["rmse_vy"],
                nom_results["rmse_total_vxy"],
                nom_results["mae_vx"],
                nom_results["mae_vy"],
                #nom_results["mae_vz"],
                nom_results["mae_total_vxy"],
                nom_results["cumulative_mae_vx"],
                nom_results["cumulative_mae_vy"],
                #nom_results["cumulative_mae_vz"],
                nom_results["cumulative_mae_total_vxy"],
                nom_results["survived_timesteps"]
            ])

            # Policy
            writer.writerow([
                "Policy vs Cmd",
                pol_results["rmse_vx"],
                pol_results["rmse_vy"],
                pol_results["rmse_total_vxy"],
                pol_results["mae_vx"],
                pol_results["mae_vy"],
                #pol_results["mae_vz"],
                pol_results["mae_total_vxy"],
                pol_results["cumulative_mae_vx"],
                pol_results["cumulative_mae_vy"],
                #pol_results["cumulative_mae_vz"],
                pol_results["cumulative_mae_total_vxy"],
                pol_results["survived_timesteps"]
            ])

            writer.writerow([])
            writer.writerow(["Z velocity RMSE (MPC vs Policy)", rmse_z])
            #writer.writerow(["Z velocity MAE (MPC vs Policy)", mae_z])

        print(f"Commanded vel + Z-vel comparison saved → {save_json_path}, {save_csv_path}")
        return all_results

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

from compareConf import policy, ep, comb, sim_time, multi, view
VIEW = view

# if multi:
#     for comb_i in comb:
#         if policy == "all":
#             # Run MPC
#             PerfRec = PerformanceRecorder([comb_i], sim_time, policy, ep)
#             PerfRec.maybe_runMPC()
#             PerfRec.compute_mpc_metrics()
            
#             # Run Policies
#             for i in ["081059", "131519", "131330", "101759"]:
#                 print(i)
#                 PerfRec = PerformanceRecorder([comb_i], sim_time, i, ep)
#                 PerfRec.maybe_runPolicy(i)
#                 PerfRec.compare_metrics()
#         else:    
#             PerfRec = PerformanceRecorder([comb_i], sim_time, policy, ep)
#             PerfRec.maybe_runMPC()
#             PerfRec.maybe_runPolicy(policy)
            
#             PerfRec.compute_mpc_metrics()
#             PerfRec.compare_metrics()
# else: 
#     if policy == "all":
#         # Run MPC
#         PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
#         PerfRec.maybe_runMPC()
#         PerfRec.compute_mpc_metrics()
        
#         # Run Policies
#         for i in ["081059", "131519", "131330", "101759"]:
#             print(i)
#             PerfRec = PerformanceRecorder(comb, sim_time, i, ep)
#             PerfRec.maybe_runPolicy(i)
#             PerfRec.compare_metrics()
#     else:    
#         PerfRec = PerformanceRecorder(comb, sim_time, policy, ep)
#         PerfRec.maybe_runPolicy(policy)
#         PerfRec.maybe_runMPC()
        
#         PerfRec.compute_mpc_metrics()
#         PerfRec.compare_metrics()


# mi sta che sta roba lho scritta solo per fare le heatmap cioe passando lintero vettore di comb
try:
    data = np.load("temp/combindex.npz")
    combindex = data['combindex']
except FileNotFoundError:
    combindex = 0  

iter = 0
combindex = int(combindex)

while combindex < len(comb) and iter < 1:
    #if iter == 5: break
    PerfRec = PerformanceRecorder([comb[combindex]], sim_time, policy, ep)
    #PerfRec.runMPC()
    PerfRec.maybe_runMPC()
    PerfRec.compute_mpc_metrics()
    for i in ["081059","131519","131330"]:
    #for i in ["111351"]:
        print(i)
        PerfRec = PerformanceRecorder([comb[combindex]], sim_time, i, ep)
        #PerfRec.rep(i)
        PerfRec.maybe_runPolicy(i)
        PerfRec.compare_metrics()
        del PerfRec
            
    combindex += 1
    iter += 1
    np.savez("temp/combindex", combindex=combindex)
