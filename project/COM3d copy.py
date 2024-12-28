import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
import pickle
import json

class TrajectoryAnalysis:
    def __init__(self, folder, policy, comb, time):
        self.folder = folder
        self.policy = policy
        self.comb = str(comb)
        self.time = str(time)
        self.ID = self.policy + self.comb + self.time  # Unique ID for naming

        self.file_policy = f"{folder}{policy}{comb}{time}.h5"
        self.file_nominal = f"{folder}Nom{comb}{time}.h5"
        
        self.com_nom = None
        self.com_policy = None
        self.cnt_nom = None
        self.cnt_policy = None

    def load_data(self):
        """Loads the nominal and policy data from HDF5 files."""
        with h5py.File(self.file_nominal, 'r') as f_nom, h5py.File(self.file_policy, 'r') as f_policy:
            # Load datasets (assuming 'states' and 'cnt' exist)
            states_nom = f_nom['states'][:]
            states_policy = f_policy['states'][:]

            if 'cnt' in f_nom and 'cnt' in f_policy:
                cnt_nom = f_nom['cnt'][:]
                cnt_policy = f_policy['cnt'][:]
            else:
                cnt_nom = None
                cnt_policy = None

        # Ensure both datasets have the same length
        min_length = min(len(states_nom), len(states_policy))
        states_nom = states_nom[:min_length]
        states_policy = states_policy[:min_length]

        if cnt_nom is not None and cnt_policy is not None:
            cnt_nom = cnt_nom[:min_length]
            cnt_policy = cnt_policy[:min_length]

        # Extract CoM positions (first three columns)
        self.com_nom = states_nom[:, :3]  # Nominal CoM positions
        self.com_policy = states_policy[:, :3]  # Policy CoM positions

        # Store the cnt arrays if available
        self.cnt_nom = cnt_nom
        self.cnt_policy = cnt_policy

    def plot_3d_trajectory(self, save_path=None):
        """Plots the 3D CoM trajectory for nominal and policy data, saving results with a structured naming system."""
        if self.com_nom is None or self.com_policy is None:
            raise ValueError("Data not loaded. Call 'load_data()' first.")

        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        # Create the subfolder for the current policy
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        pkl_file = os.path.join(policy_folder, f"3D_CoM_trajectory_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"3D_CoM_trajectory_{self.ID}.png")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot MPC CoM trajectory
        ax.plot(self.com_nom[:, 0], self.com_nom[:, 1], self.com_nom[:, 2], 
                label='MPC CoM Trajectory', color='blue')

        # Plot Policy CoM trajectory
        ax.plot(self.com_policy[:, 0], self.com_policy[:, 1], self.com_policy[:, 2], 
                label='Policy CoM Trajectory', color='red')

        ax.set_xlabel('CoM X-Position')
        ax.set_ylabel('CoM Y-Position')
        ax.set_zlabel('CoM Z-Position')
        ax.set_title('3D CoM Trajectory Comparison')
        ax.legend()

        # Adjust aspect ratio if desired
        ax.set_box_aspect([
            np.ptp(self.com_nom[:, 0]), 
            np.ptp(self.com_nom[:, 1]), 
            np.ptp(self.com_nom[:, 2])
        ])

        # Save the figure as interactive .pkl
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to {pkl_file}")

        # Save as PNG
        plt.savefig(png_file)
        print(f"Saved static image to {png_file}")

        plt.show()

    def plot_ee_trajectories_3d_separate(self, save_path=None):
        """Plots each end-effector trajectory in separate 3D figures with boolean flags indicated."""
        if self.cnt_nom is None or self.cnt_policy is None:
            raise ValueError("CNT data not loaded or not available. Ensure 'cnt' dataset is present and call 'load_data()' first.")

        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # Indices for each EE
        ee_indices = [
            (0, 1, 2, 3),    # EE0
            (4, 5, 6, 7),    # EE1
            (8, 9, 10, 11),  # EE2
            (12, 13, 14, 15) # EE3
        ]

        for i, (b_idx, x_idx, y_idx, z_idx) in enumerate(ee_indices):
            bool_nom = self.cnt_nom[:, b_idx].astype(bool)
            x_nom, y_nom, z_nom = self.cnt_nom[:, x_idx], self.cnt_nom[:, y_idx], self.cnt_nom[:, z_idx]

            bool_pol = self.cnt_policy[:, b_idx].astype(bool)
            x_pol, y_pol, z_pol = self.cnt_policy[:, x_idx], self.cnt_policy[:, y_idx], self.cnt_policy[:, z_idx]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot nominal MPC trajectory with bool differentiation
            ax.scatter(x_nom[bool_nom], y_nom[bool_nom], z_nom[bool_nom],
                       color='blue', label=f'EE{i} MPC (bool=1)', marker='o')
            ax.scatter(x_nom[~bool_nom], y_nom[~bool_nom], z_nom[~bool_nom],
                       color='gray', label=f'EE{i} MPC (bool=0)', marker='x')

            # Plot policy trajectory with bool differentiation
            ax.scatter(x_pol[bool_pol], y_pol[bool_pol], z_pol[bool_pol],
                       color='red', label=f'EE{i} Policy (bool=1)', marker='o')
            ax.scatter(x_pol[~bool_pol], y_pol[~bool_pol], z_pol[~bool_pol],
                       color='black', label=f'EE{i} Policy (bool=0)', marker='x')

            # Set equal axes
            all_x = np.concatenate([x_nom, x_pol])
            all_y = np.concatenate([y_nom, y_pol])
            all_z = np.concatenate([z_nom, z_pol])

            max_range = np.max([np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)]) / 2.0
            mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel('X-Position')
            ax.set_ylabel('Y-Position')
            ax.set_zlabel('Z-Position')
            ax.set_title(f'3D Trajectory of EE{i} (bool flag shown)')
            ax.legend()

            # Save with ID
            pkl_file = os.path.join(policy_folder, f"EE{i}_trajectory_{self.ID}.pkl")
            png_file = os.path.join(policy_folder, f"EE{i}_trajectory_bool_{self.ID}.png")

            with open(pkl_file, 'wb') as f:
                pickle.dump(fig, f)
            print(f"Saved interactive figure EE{i} to {pkl_file}")

            plt.savefig(png_file)
            print(f"Saved static image EE{i} to {png_file}")

            plt.show()

    def plot_euclidean_distance(self, save_path=None):
        """Plots the Euclidean distance between nominal and policy trajectories over time."""
        if self.com_nom is None or self.com_policy is None:
            raise ValueError("Data not loaded. Call 'load_data()' first.")

        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # Compute Euclidean distance
        distances = np.linalg.norm(self.com_nom - self.com_policy, axis=1)

        fig = plt.figure()
        plt.plot(distances, label='Euclidean Distance Over Time', color='green')
        plt.xlabel('Time Steps')
        plt.ylabel('Distance')
        plt.title('Trajectory Divergence Over Time')
        plt.legend()
        plt.grid()

        # Save with ID
        pkl_file = os.path.join(policy_folder, f"Euclidean_distance_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"Euclidean_distance_{self.ID}.png")

        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to {pkl_file}")

        plt.savefig(png_file)
        print(f"Saved static image to {png_file}")

        plt.show()

    def plot_from_json_with_labels(self, json_path, save_dir):
        """
        Loads JSON data and generates labeled scatter plots for different variable categories.
        Uses the same naming convention and directory structure.
        """
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Load JSON data
        with open(json_path, "r") as file:
            data = json.load(file)

        # Extract relevant data
        rmse_states = data.get("rmse_states_per_variable", [])
        rmse_velocities = data.get("rmse_velocities_per_variable", [])

        # Define labels for different variable categories
        # (We don't necessarily need these arrays since we split them by known indexing)
        state_labels = ["x", "y", "z", "qx", "qy", "qz", "qw"] + [f"q{i}" for i in range(1, len(rmse_states) - 6 + 1)]
        velocity_labels = ["vx", "vy", "vz", "wx", "wy", "wz"] + [f"v{i}" for i in range(1, len(rmse_velocities) - 6 + 1)]

        # Split state variables into categories
        base_states = rmse_states[:3]  # x, y, z
        quaternion_states = rmse_states[3:7]  # qx, qy, qz, qw
        joint_states = rmse_states[7:]  # q1, q2, ...

        # Split velocity variables into categories
        base_velocities = rmse_velocities[:3]  # vx, vy, vz
        angular_velocities = rmse_velocities[3:6]  # wx, wy, wz
        joint_velocities = rmse_velocities[6:]  # v1, v2, ...

        # Plot base position RMSE
        plt.figure()
        plt.scatter(range(len(base_states)), base_states, label="Base Position RMSE")
        plt.xticks(range(len(base_states)), ["x", "y", "z"])
        plt.ylabel("RMSE")
        plt.title("Base Position RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("base_position_rmse_" + self.ID + ".png")))
        plt.close()

        # Plot quaternion RMSE
        plt.figure()
        plt.scatter(range(len(quaternion_states)), quaternion_states, label="Quaternion RMSE")
        plt.xticks(range(len(quaternion_states)), ["qx", "qy", "qz", "qw"])
        plt.ylabel("RMSE")
        plt.title("Quaternion RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("quaternion_rmse_" + self.ID + ".png")))
        plt.close()

        # Plot joint position RMSE
        plt.figure()
        plt.scatter(range(len(joint_states)), joint_states, label="Joint Position RMSE")
        plt.xticks(range(len(joint_states)), [f"q{i+1}" for i in range(len(joint_states))], rotation=90)
        plt.ylabel("RMSE")
        plt.title("Joint Position RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("joint_position_rmse_" + self.ID + ".png")))
        plt.close()

        # Plot base linear velocity RMSE
        plt.figure()
        plt.scatter(range(len(base_velocities)), base_velocities, label="Base Linear Velocity RMSE")
        plt.xticks(range(len(base_velocities)), ["vx", "vy", "vz"])
        plt.ylabel("RMSE")
        plt.title("Base Linear Velocity RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("base_linear_velocity_rmse_" + self.ID + ".png")))
        plt.close()

        # Plot angular velocity RMSE
        plt.figure()
        plt.scatter(range(len(angular_velocities)), angular_velocities, label="Angular Velocity RMSE")
        plt.xticks(range(len(angular_velocities)), ["wx", "wy", "wz"])
        plt.ylabel("RMSE")
        plt.title("Angular Velocity RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("angular_velocity_rmse_" + self.ID + ".png")))
        plt.close()

        # Plot joint velocity RMSE
        plt.figure()
        plt.scatter(range(len(joint_velocities)), joint_velocities, label="Joint Velocity RMSE")
        plt.xticks(range(len(joint_velocities)), [f"v{i+1}" for i in range(len(joint_velocities))], rotation=90)
        plt.ylabel("RMSE")
        plt.title("Joint Velocity RMSE")
        plt.legend()
        plt.savefig(os.path.join(save_dir, ("joint_velocity_rmse_" + self.ID + ".png")))
        plt.close()

        print(f"Plots saved in {save_dir}")
##########################################################
#  Adjust these to point to your actual code & variables #
##########################################################
from compareConf import policy, ep, comb, sim_time, multi
import os

folder = "/home/federico/biconmp_mujoco/project/results/tracking_perf_"
save_dir = "/home/federico/biconmp_mujoco/project/results/plots"

# The same set of policies you do multiple runs for when policy == 'all'
ALL_POLICIES = ["081059", "131519", "131330"]

def run_trajectory_analysis_multi(combinations, sim_time, selected_policy, multi_flag):
    """
    Runs TrajectoryAnalysis in a loop, once per sub-combination, 
    and for each policy if 'selected_policy' == 'all'.
    """

    # Utility: if user gave a single combination as a list (e.g. [0, 0.3, 0, 0]) 
    # but multi_flag is True, you can decide if you want to wrap it in a list.
    # Or just assume 'combinations' is always a list of combos.

    if multi_flag:
        # We iterate over each item in 'combinations'
        for single_comb in combinations:
            # If policy == "all", do all known policies
            if selected_policy == "all":
                for pol in ALL_POLICIES:
                    print(f"[MULTI:True] Running comb={single_comb}, policy={pol}")

                    # Instantiate with one sub-combo
                    analysis = TrajectoryAnalysis(
                        folder=folder,
                        policy=pol,
                        comb=single_comb,  # e.g. [0, 0.3, 0, 0]
                        time=sim_time
                    )

                    # Load and do your normal plotting
                    analysis.load_data()
                    analysis.plot_3d_trajectory(save_dir)
                    analysis.plot_ee_trajectories_3d_separate(save_dir)
                    analysis.plot_euclidean_distance(save_dir)

                    # If you also want the JSON-based plots
                    json_path = os.path.join(
                        folder.rsplit("/", 1)[0],  # => "/home/federico/biconmp_mujoco/project/results"
                        f"tracking_perf_{analysis.ID}_comparison_results.json"
                    )
                    analysis.plot_from_json_with_labels(json_path, save_dir)
            
            # else if single policy
            else:
                print(f"[MULTI:True] Running comb={single_comb}, policy={selected_policy}")
                analysis = TrajectoryAnalysis(
                    folder=folder,
                    policy=selected_policy,
                    comb=single_comb,
                    time=sim_time
                )

                analysis.load_data()
                analysis.plot_3d_trajectory(save_dir)
                analysis.plot_ee_trajectories_3d_separate(save_dir)
                analysis.plot_euclidean_distance(save_dir)

                json_path = os.path.join(
                    folder.rsplit("/", 1)[0],
                    f"tracking_perf_{analysis.ID}_comparison_results.json"
                )
                analysis.plot_from_json_with_labels(json_path, save_dir)
    else:
        # multi_flag == False => we assume 'combinations' is either a single combo 
        # or you just want to do one iteration, similarly:
        single_comb = combinations  # Possibly a single list, e.g. [0, 0.3, 0, 0]

        if selected_policy == "all":
            for pol in ALL_POLICIES:
                print(f"[MULTI:False] Running single comb={single_comb}, policy={pol}")
                analysis = TrajectoryAnalysis(folder, pol, single_comb, sim_time)

                analysis.load_data()
                analysis.plot_3d_trajectory(save_dir)
                analysis.plot_ee_trajectories_3d_separate(save_dir)
                analysis.plot_euclidean_distance(save_dir)

                json_path = os.path.join(
                    folder.rsplit("/", 1)[0],
                    f"tracking_perf_{analysis.ID}_comparison_results.json"
                )
                analysis.plot_from_json_with_labels(json_path, save_dir)
        else:
            print(f"[MULTI:False] Running single comb={single_comb}, policy={selected_policy}")
            analysis = TrajectoryAnalysis(folder, selected_policy, single_comb, sim_time)

            analysis.load_data()
            analysis.plot_3d_trajectory(save_dir)
            analysis.plot_ee_trajectories_3d_separate(save_dir)
            analysis.plot_euclidean_distance(save_dir)

            json_path = os.path.join(
                folder.rsplit("/", 1)[0],
                f"tracking_perf_{analysis.ID}_comparison_results.json"
            )
            analysis.plot_from_json_with_labels(json_path, save_dir)


run_trajectory_analysis_multi(
combinations=comb,        # from compareConf
sim_time=sim_time,        # from compareConf
selected_policy=policy,   # from compareConf
multi_flag=multi          # from compareConf
)
