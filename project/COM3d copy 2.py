import ast
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
from mpl_toolkits.mplot3d import proj3d
import pickle
import json


##############################################################################
# 1) Add a dictionary mapping your policy IDs to descriptive labels
##############################################################################
policy_labels = {
    "081059": "081059: Hybrid",
    "131519": "131519: Velocity Conditioned",
    "131330": "131330: Contact Conditioned"
}

def get_policy_label(policy_id: str) -> str:
    """Return the descriptive label for a given policy_id."""
    return policy_labels.get(policy_id, policy_id)

class TrajectoryAnalysis:
    def __init__(self, folder, policy, comb, time):
        self.folder = folder
        self.policy = policy
        self.comb = str(comb)
        self.time = str(time)

        # If policy == "all", we will handle multiple policies
        if self.policy == "all":
            self.policies_to_compare = ["081059", "131519", "131330"]
            self.ID = "all" + self.comb + self.time
        else:
            self.policies_to_compare = [self.policy]
            self.ID = self.policy + self.comb + self.time

        # Nominal file
        self.file_nominal = f"{folder}Nom{self.comb}{self.time}.h5"

        # Data structures to hold loaded data
        self.com_nom = None
        self.cnt_nom = None

        # For multiple policies, we store data in dictionaries keyed by policy name
        self.com_policies = {}
        self.cnt_policies = {}

    def load_data(self):
        """Loads the nominal and policy data from HDF5 files, padding shorter trajectories with the last value."""
        with h5py.File(self.file_nominal, 'r') as f_nom:
            states_nom = f_nom['states'][:]              # Existing CoM data
            cnt_nom = f_nom['cnt'][:] if 'cnt' in f_nom else None
            
            # --- NEW: Load velocities (assuming shape NxD, with D>=2, the first two are vx, vy) ---
            if 'v' in f_nom:
                v_nom = f_nom['v'][:]
            else:
                raise ValueError(f"No 'v' dataset in nominal file {self.file_nominal}")

        # Store nominal CoM and contact points
        self.com_nom = states_nom[:, :3]  # unchanged
        self.cnt_nom = cnt_nom
        # Store nominal velocities (just the first two columns: vx, vy)
        self.vel_nom = v_nom[:, 0:2]

        # Store the true length of the nominal trajectory
        self.true_nom_len = len(self.com_nom)

        # Find the maximum length across all policies
        max_length = self.true_nom_len

        # Prepare dicts to store policy data
        self.true_policy_lens = {}
        self.com_policies = {}
        self.cnt_policies = {}
        self.vel_policies = {}  # <--- NEW dictionary to hold velocities

        # Load each policy data
        for pol in self.policies_to_compare:
            file_policy = f"{self.folder}{pol}{self.comb}{self.time}.h5"
            with h5py.File(file_policy, 'r') as f_policy:
                states_policy = f_policy['states'][:]
                cnt_policy = f_policy['cnt'][:] if 'cnt' in f_policy else None
                
                if 'v' in f_policy:
                    v_policy = f_policy['v'][:]
                    # breakpoint()
                else:
                    raise ValueError(f"No 'v' dataset in policy file {file_policy}")
            # Store true length
            self.true_policy_lens[pol] = len(states_policy)
            max_length = max(max_length, len(states_policy))

            # Store data
            self.com_policies[pol] = states_policy[:, :3]
            self.cnt_policies[pol] = cnt_policy
            self.vel_policies[pol] = v_policy[:, 0:2]  # store only (vx, vy)

        # breakpoint()
        # Pad function
        def pad_with_last_value(data, target_length):
            padded = np.full((target_length, data.shape[1]), data[-1])  # Fill with last row
            padded[:len(data)] = data
            return padded

        # Pad nominal
        self.com_nom = pad_with_last_value(self.com_nom, max_length)
        self.vel_nom = pad_with_last_value(self.vel_nom, max_length)
        if self.cnt_nom is not None:
            self.cnt_nom = pad_with_last_value(self.cnt_nom, max_length)

        # Pad policies
        for pol in self.policies_to_compare:
            self.com_policies[pol] = pad_with_last_value(self.com_policies[pol], max_length)
            self.vel_policies[pol] = pad_with_last_value(self.vel_policies[pol], max_length)
            if self.cnt_policies[pol] is not None:
                self.cnt_policies[pol] = pad_with_last_value(self.cnt_policies[pol], max_length)

    def plot_3d_trajectory(self, save_path=None):
        """Plots the 3D CoM trajectory for nominal and policy data,
        using auto-ticks but hiding numeric labels.
        Adds a note about approximate grid spacing."""
        if self.com_nom is None or len(self.com_policies) == 0:
            raise ValueError("Data not loaded. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        pkl_file = os.path.join(policy_folder, f"3D_CoM_trajectory_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"3D_CoM_trajectory_{self.ID}.png")

        fig = plt.figure(figsize=(18, 12), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # -- Plot the nominal (MPC) trajectory --
        ax.plot(
            self.com_nom[:, 0],
            self.com_nom[:, 1],
            self.com_nom[:, 2],
            label='MPC CoM Trajectory',
            color='blue',
            linewidth=2
        )

        # -- Plot each policy --
        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            mapped_label = get_policy_label(pol)  # e.g. "Hybrid"
            ax.plot(
                self.com_policies[pol][:, 0],
                self.com_policies[pol][:, 1],
                self.com_policies[pol][:, 2],
                label=f'{mapped_label} CoM Trajectory',
                color=colors[i % len(colors)],
                linewidth=2
            )

        # Axis labels (disable auto-rotation)
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel("CoM X-Position", rotation=0, labelpad=15, fontsize=12)
        ax.set_ylabel("CoM Y-Position", rotation=0, labelpad=15, fontsize=12)
        ax.set_zlabel("CoM Z-Position", rotation=0, labelpad=15, fontsize=12)

        ax.set_title('3D CoM Trajectory Comparison', pad=20, fontsize=14)
        ax.legend(fontsize=10)

        # ----------------------------------------------------------------------
        # Let Matplotlib auto-decide the ticks, but remove the labels.
        # ----------------------------------------------------------------------
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # ----------------------------------------------------------------------
        # Extract the auto ticks & compute approximate spacing for each axis.
        # ----------------------------------------------------------------------
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        zticks = ax.get_zticks()

        def approx_spacing(t):
            if len(t) >= 2:
                return np.round(np.mean(np.diff(t)), 2)
            return None

        x_spacing = approx_spacing(xticks)
        y_spacing = approx_spacing(yticks)
        z_spacing = approx_spacing(zticks)

        # ----------------------------------------------------------------------
        # Show a small text note in the corner with the spacing
        # ----------------------------------------------------------------------
        spacing_note = []
        if x_spacing is not None:
            spacing_note.append(f"X~{x_spacing}")
        if y_spacing is not None:
            spacing_note.append(f"Y~{y_spacing}")
        if z_spacing is not None:
            spacing_note.append(f"Z~{z_spacing}")

        if spacing_note:
            text_str = "Grid spacing approx: " + ", ".join(spacing_note)
            ax.text2D(
                0.01, 0.95,
                text_str,
                transform=ax.transAxes,
                color='gray',
                fontsize=10
            )

        # ----------------------------------------------------------------------
        # Adjust aspect ratio & viewpoint
        # ----------------------------------------------------------------------
        all_x = np.concatenate([
            self.com_nom[:, 0]
        ] + [self.com_policies[pol][:, 0] for pol in self.policies_to_compare])
        all_y = np.concatenate([
            self.com_nom[:, 1]
        ] + [self.com_policies[pol][:, 1] for pol in self.policies_to_compare])
        all_z = np.concatenate([
            self.com_nom[:, 2]
        ] + [self.com_policies[pol][:, 2] for pol in self.policies_to_compare])
        ax.set_box_aspect((np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)))

        ax.view_init(elev=30, azim=-60)

        plt.tight_layout()

        # ----------------------------------------------------------------------
        # Save & show
        # ----------------------------------------------------------------------
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(png_file)
        plt.show()

    def plot_ee_trajectories_3d_separate(self, save_path=None):
        """
        Plots each end-effector trajectory in separate 3D figures with boolean flags indicated.
        If policy=='all', it plots MPC and all three policies.
        """
        if self.cnt_nom is None or any(self.cnt_policies[pol] is None for pol in self.policies_to_compare):
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
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')

            # Plot MPC
            bool_nom = self.cnt_nom[:, b_idx].astype(bool)
            x_nom, y_nom, z_nom = (
                self.cnt_nom[:, x_idx],
                self.cnt_nom[:, y_idx],
                self.cnt_nom[:, z_idx]
            )
            ax.scatter(
                x_nom[bool_nom],
                y_nom[bool_nom],
                z_nom[bool_nom],
                color='blue',
                label=f'EE{i} MPC (bool=1)',
                marker='o'
            )
            ax.scatter(
                x_nom[~bool_nom],
                y_nom[~bool_nom],
                z_nom[~bool_nom],
                color='gray',
                label=f'EE{i} MPC (bool=0)',
                marker='x'
            )

            # Colors for policies
            colors = ['red', 'green', 'orange']
            all_x = [x_nom]
            all_y = [y_nom]
            all_z = [z_nom]

            for j, pol in enumerate(self.policies_to_compare):
                cnt_policy = self.cnt_policies[pol]
                bool_pol = cnt_policy[:, b_idx].astype(bool)
                x_pol = cnt_policy[:, x_idx]
                y_pol = cnt_policy[:, y_idx]
                z_pol = cnt_policy[:, z_idx]

                all_x.append(x_pol)
                all_y.append(y_pol)
                all_z.append(z_pol)

                mapped_label = get_policy_label(pol)
                ax.scatter(
                    x_pol[bool_pol],
                    y_pol[bool_pol],
                    z_pol[bool_pol],
                    color=colors[j % len(colors)],
                    label=f'EE{i} {mapped_label} (bool=1)',
                    marker='o'
                )
                ax.scatter(
                    x_pol[~bool_pol],
                    y_pol[~bool_pol],
                    z_pol[~bool_pol],
                    color='black',
                    label=f'EE{i} {mapped_label} (bool=0)',
                    marker='x'
                )

            # Set equal axes
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            all_z = np.concatenate(all_z)

            max_range = np.max([
                np.ptp(all_x),
                np.ptp(all_y),
                np.ptp(all_z)
            ]) / 2.0
            mid_x, mid_y, mid_z = (
                np.mean(all_x),
                np.mean(all_y),
                np.mean(all_z)
            )
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Formatting
            ax.set_xlabel('X-Position', labelpad=12)
            ax.set_ylabel('Y-Position', labelpad=12)
            ax.set_zlabel('Z-Position', labelpad=12)
            ax.set_title(f'3D Trajectory of EE{i} (bool flag shown)', pad=20)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.tick_params(axis='z', labelsize=10)

            ax.view_init(elev=30, azim=-60)

            ax.legend()
            plt.tight_layout()

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
        """
        Plots the Euclidean distance between nominal and policy trajectories over time,
        ensuring that we compute distances only for the actual (unpadded) length of the
        nominal trajectory. Any padded data beyond the nominal trajectory length is ignored.
        """
        if self.com_nom is None or len(self.com_policies) == 0:
            raise ValueError("Data not loaded. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Distinguish policies by color for clarity
        colors = ['red', 'green', 'orange']

        for i, pol in enumerate(self.policies_to_compare):
            # True lengths of the nominal and policy trajectories
            nominal_len = self.true_nom_len
            policy_len = self.true_policy_lens[pol]

            # Compute the effective comparison length
            length = min(nominal_len, policy_len)

            # Compute Euclidean distance up to the actual trajectory length
            distances = np.linalg.norm(
                self.com_nom[:length] - self.com_policies[pol][:length],
                axis=1
            )

            mapped_label = get_policy_label(pol)
            ax.plot(
                distances, 
                label=f'Euclidean Distance {mapped_label}',
                color=colors[i % len(colors)]
            )

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Distance')
        ax.set_title('Trajectory Divergence Over Time')
        ax.legend()
        ax.grid(True)

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
        state_labels = ["x", "y", "z", "qx", "qy", "qz", "qw"] + [
            f"q{i}" for i in range(1, len(rmse_states) - 6 + 1)
        ]
        velocity_labels = ["vx", "vy", "vz", "wx", "wy", "wz"] + [
            f"v{i}" for i in range(1, len(rmse_velocities) - 6 + 1)
        ]

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
        
    def plot_2d_trajectory(self, save_path=None):
        """
        Plots a 2D CoM trajectory (X vs Y) for the nominal and policy data,
        ignoring the Z dimension. Similar to 'plot_3d_trajectory' but in 2D.
        - Removes numeric tick labels while letting Matplotlib auto-decide them.
        - Shows approximate grid spacing (like the 3D version).
        - Places a marker (circle) each time the controller switches:
        if sim_time=1.3, we mark at 1300, 2600, 3900... up to the trajectory length.
        - Maintains equal aspect ratio.
        - Ensures the legend is fully visible within the saved figure.
        """
        if self.com_nom is None or len(self.policies_to_compare) == 0:
            raise ValueError("Data not loaded. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        pkl_file = os.path.join(policy_folder, f"2D_CoM_trajectory_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"2D_CoM_trajectory_{self.ID}.png")

        fig, ax = plt.subplots(figsize=(9, 9))

        # Determine all switch indices (multiples of switch_step)
        sim_t = float(self.time)  # Convert self.time (string) to float
        switch_step = int(sim_t * 1000)  # e.g. for time=1.3s => switch_step=1300

        # Find the maximum trajectory length
        max_length = max([len(self.com_nom)] + [len(self.com_policies[pol]) for pol in self.policies_to_compare])

        # Pad the nominal trajectory with NaN if it is shorter
        com_nom_padded = np.full((max_length, 2), np.nan)
        com_nom_padded[:len(self.com_nom), :] = self.com_nom[:, :2]

        # Plot the nominal trajectory (X vs Y)
        ax.plot(
            com_nom_padded[:, 0],
            com_nom_padded[:, 1],
            label='MPC CoM Trajectory',
            color='blue',
            linewidth=2
        )

        # Mark a circle at each switch multiple for the nominal trajectory
        switch_indices_nom = range(switch_step, len(com_nom_padded), switch_step)
        for sw_idx in switch_indices_nom:
            if sw_idx < len(com_nom_padded):
                ax.plot(
                    com_nom_padded[sw_idx, 0],
                    com_nom_padded[sw_idx, 1],
                    marker='o',
                    markersize=10,
                    color='blue',
                    label='MPC Switch Point' if sw_idx == switch_step else None
                )

        # Plot each policy trajectory & mark switches
        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            com_policy_padded = np.full((max_length, 2), np.nan)
            com_policy_padded[:len(self.com_policies[pol]), :] = self.com_policies[pol][:, :2]

            mapped_label = get_policy_label(pol)
            ax.plot(
                com_policy_padded[:, 0],
                com_policy_padded[:, 1],
                label=f'{mapped_label} CoM Trajectory',
                color=colors[i % len(colors)],
                linewidth=2
            )

            # Mark a circle at each switch multiple
            switch_indices_pol = range(switch_step, len(com_policy_padded), switch_step)
            for sw_idx in switch_indices_pol:
                if sw_idx < len(com_policy_padded):
                    ax.plot(
                        com_policy_padded[sw_idx, 0],
                        com_policy_padded[sw_idx, 1],
                        marker='o',
                        markersize=10,
                        color=colors[i % len(colors)],
                        label=f'{mapped_label} Switch Point' if sw_idx == switch_step else None
                    )

        # Adjust axis limits based on the data range
        all_x = np.concatenate([
            com_nom_padded[:, 0],
            *(self.com_policies[pol][:, 0] for pol in self.policies_to_compare)
        ])
        all_y = np.concatenate([
            com_nom_padded[:, 1],
            *(self.com_policies[pol][:, 1] for pol in self.policies_to_compare)
        ])

        # Remove NaNs
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]

        # Set axis limits based on the full range of data
        ax.set_xlim(all_x.min() - 0.01, all_x.max() + 0.01)
        ax.set_ylim(all_y.min() - 0.01, all_y.max() + 0.01)

        # **Set equal aspect ratio**
        ax.set_aspect('equal', adjustable='box')

        # Formatting
        ax.set_xlabel("CoM X-Position", fontsize=12)
        ax.set_ylabel("CoM Y-Position", fontsize=12)
        ax.set_title("2D CoM Trajectory Comparison", fontsize=14)
        ax.legend(fontsize=10, loc='best')  # Position legend inside plot

        # Remove numeric tick labels but keep tick marks
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])

        # Adjust layout to make room for the legend
        plt.tight_layout()

        # Save figure in .pkl (interactive) and .png with tight bounding box
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(png_file)  # Ensure legend is included
        plt.show()

    def plot_ee_trajectories_2d_separate_contact_only(self, save_path=None):
        """
        Plots the X-Y trajectory of each end-effector in separate 2D figures,
        showing ONLY the points where the EE is in contact (bool=1).
        Z dimension is ignored (projection to X-Y plane).
        """
        if self.cnt_nom is None or any(self.cnt_policies[pol] is None for pol in self.policies_to_compare):
            raise ValueError("CNT data not loaded or not available. Ensure 'cnt' dataset is present and call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # Indices for each EE
        ee_indices = [
            (0, 1, 2, 3),    # EE0: (bool, x, y, z)
            (4, 5, 6, 7),    # EE1
            (8, 9, 10, 11),  # EE2
            (12, 13, 14, 15) # EE3
        ]

        for i, (b_idx, x_idx, y_idx, z_idx) in enumerate(ee_indices):
            fig, ax = plt.subplots(figsize=(14, 5))

            # Plot MPC CONTACT ONLY
            bool_nom = self.cnt_nom[:, b_idx].astype(bool)
            x_nom = self.cnt_nom[:, x_idx]
            y_nom = self.cnt_nom[:, y_idx]

            # Filter by contact
            x_nom_contact = x_nom[bool_nom]
            y_nom_contact = y_nom[bool_nom]

            ax.scatter(
                x_nom_contact,
                y_nom_contact,
                color='blue',
                label=f'EE{i} MPC (contact only)',
                marker='o',
                s=1
            )

            # Colors for policies
            colors = ['red', 'green', 'orange']
            all_x = [x_nom_contact]
            all_y = [y_nom_contact]

            for j, pol in enumerate(self.policies_to_compare):
                cnt_policy = self.cnt_policies[pol]
                bool_pol = cnt_policy[:, b_idx].astype(bool)
                x_pol = cnt_policy[:, x_idx]
                y_pol = cnt_policy[:, y_idx]

                # Only contact points
                x_pol_contact = x_pol[bool_pol]
                y_pol_contact = y_pol[bool_pol]

                all_x.append(x_pol_contact)
                all_y.append(y_pol_contact)

                mapped_label = get_policy_label(pol)
                ax.scatter(
                    x_pol_contact,
                    y_pol_contact,
                    color=colors[j % len(colors)],
                    label=f'EE{i} {mapped_label} (contact only)',
                    marker='o',
                    s=1
                )

            # Adjust aspect ratio
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)
            ax.set_aspect('equal', 'box')
            ax.set_xlim([all_x.min(), all_x.max()])
            ax.set_ylim([all_y.min(), all_y.max()])

            ax.set_xlabel('X-Position', fontsize=12)
            ax.set_ylabel('Y-Position', fontsize=12)
            ax.set_title(f'2D Contact-Only Trajectory of EE{i}', fontsize=14)
            ax.legend(fontsize=10)

            # Save with ID
            pkl_file = os.path.join(policy_folder, f"EE{i}_trajectory_2D_contact_{self.ID}.pkl")
            png_file = os.path.join(policy_folder, f"EE{i}_trajectory_2D_contact_{self.ID}.png")

            with open(pkl_file, 'wb') as f:
                pickle.dump(fig, f)
            print(f"Saved interactive 2D figure EE{i} to {pkl_file}")

            plt.savefig(png_file)
            print(f"Saved static 2D image EE{i} to {png_file}")

            plt.show()

    def plot_2d_trajectory_xy_and_xz(self, save_path=None):
        """
        Plots the CoM trajectory in 2D from two different views (vertical layout):
        - Top subplot: Top view (X vs Y)
        - Bottom subplot: Side view (X vs Z)
        """
        if self.com_nom is None or len(self.com_policies) == 0:
            raise ValueError("Data not loaded. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        # Create directory if it doesn't exist
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # File names
        pkl_file = os.path.join(policy_folder, f"2D_CoM_trajectory_xy_xz_vertical_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"2D_CoM_trajectory_xy_xz_vertical_{self.ID}.png")

        # Prepare figure with 2 subplots in vertical alignment (2 rows, 1 column)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 10), dpi=100)
        
        
        ax_top = axes[0]   # X vs Y
        ax_side = axes[1]  # X vs Z

        colors = ['red', 'green', 'orange']

        # --------------------------------------------------------------------------
        # 1) TOP VIEW (X vs Y)
        # --------------------------------------------------------------------------
        ax_top.plot(
            self.com_nom[:, 0],
            self.com_nom[:, 1],
            label='MPC CoM (top view)',
            color='blue',
            linewidth=2
        )

        for i, pol in enumerate(self.policies_to_compare):
            mapped_label = get_policy_label(pol)
            ax_top.plot(
                self.com_policies[pol][:, 0],
                self.com_policies[pol][:, 1],
                label=f'{mapped_label} (top view)',
                color=colors[i % len(colors)],
                linewidth=2
            )

        ax_top.set_xlabel("X-Position", fontsize=12)
        ax_top.set_ylabel("Y-Position", fontsize=12)
        ax_top.set_title("Top View (X vs Y)", fontsize=14)
        ax_top.set_aspect('equal', 'box')
        ax_top.legend(fontsize=9)

        # --------------------------------------------------------------------------
        # 2) SIDE VIEW (X vs Z)
        # --------------------------------------------------------------------------
        ax_side.plot(
            self.com_nom[:, 0],
            self.com_nom[:, 2],
            label='MPC CoM (side view)',
            color='blue',
            linewidth=2
        )

        for i, pol in enumerate(self.policies_to_compare):
            mapped_label = get_policy_label(pol)
            ax_side.plot(
                self.com_policies[pol][:, 0],
                self.com_policies[pol][:, 2],
                label=f'{mapped_label} (side view)',
                color=colors[i % len(colors)],
                linewidth=2
            )

        ax_side.set_xlabel("X-Position", fontsize=12)
        ax_side.set_ylabel("Z-Position", fontsize=12)
        ax_side.set_title("Side View (X vs Z)", fontsize=14)
        ax_side.set_aspect('equal', 'box')
        ax_side.legend(fontsize=9)

        # Layout and saving
        plt.tight_layout()
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(png_file)
        plt.show()

    def plot_com_z_position(self, save_path=None):
        """
        Plots the Z-position (height) of the CoM over time for:
        - The nominal (MPC) trajectory
        - Each policy trajectory
        """
        # Make sure we have data loaded
        if self.com_nom is None or len(self.com_policies) == 0:
            raise ValueError("Data not loaded. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # Build filenames for pickled figure and PNG
        pkl_file = os.path.join(policy_folder, f"CoM_Z_position_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"CoM_Z_position_{self.ID}.png")

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

        # --- Plot Nominal (MPC) CoM Z ---
        ax.plot(
            self.com_nom[:, 2],          # Z is the 3rd column (index 2)
            label='MPC CoM Z-Position',
            color='blue',
            linewidth=2
        )

        # --- Plot each policy's CoM Z ---
        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            mapped_label = get_policy_label(pol)
            ax.plot(
                self.com_policies[pol][:, 2],  # Z is again index 2
                label=f'{mapped_label} CoM Z-Position',
                color=colors[i % len(colors)],
                linewidth=2
            )

        # Style & Labels
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Z-Position', fontsize=12)
        ax.set_title('CoM Z-Position Over Time', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)

        plt.tight_layout()

        # Save figure in .pkl (interactive) and .png
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(png_file)
        plt.show()

        print(f"Saved interactive figure to: {pkl_file}")
        print(f"Saved static image to: {png_file}")

    def plot_combined_2d_trajectories(self, save_path=None):
        """
        Combines the 2D CoM trajectory with the 2D contact-only end-effector trajectories
        into a single figure. Only includes EE0. Adds switching point markers.
        - Maintains equal aspect ratio.
        - Ensures the legend is fully visible within the saved figure.
        """
        if self.com_nom is None or len(self.policies_to_compare) == 0:
            raise ValueError("CoM data not loaded. Call 'load_data()' first.")
        if self.cnt_nom is None or any(self.cnt_policies[pol] is None for pol in self.policies_to_compare):
            raise ValueError("CNT data not loaded or not available. Call 'load_data()' first.")
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")

        # Determine switching interval (samples)
        sim_t = float(self.time)  # Convert self.time (string) to float
        switch_step = int(sim_t * 1000)  # e.g., for sim_time = 1.3, switch_step = 1300

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # File names for saving
        pkl_file = os.path.join(policy_folder, f"Combined_2D_trajectories_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"Combined_2D_trajectories_{self.ID}.png")

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

        # --- Plot 2D CoM Trajectory ---
        ax.plot(
            self.com_nom[:, 0],
            self.com_nom[:, 1],
            label='MPC CoM Trajectory',
            color='blue',
            linewidth=2
        )

        # Add switching point markers for nominal trajectory
        switch_indices_nom = range(switch_step, len(self.com_nom), switch_step)
        for sw_idx in switch_indices_nom:
            ax.plot(
                self.com_nom[sw_idx, 0],
                self.com_nom[sw_idx, 1],
                marker='o',
                markersize=10,
                color='blue',
                label='MPC Switch Point' if sw_idx == switch_step else None
            )

        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            mapped_label = get_policy_label(pol)
            ax.plot(
                self.com_policies[pol][:, 0],
                self.com_policies[pol][:, 1],
                label=f'{mapped_label} CoM Trajectory',
                color=colors[i % len(colors)],
                linewidth=2
            )

            # Add switching point markers for each policy trajectory
            switch_indices_pol = range(switch_step, len(self.com_policies[pol]), switch_step)
            for sw_idx in switch_indices_pol:
                ax.plot(
                    self.com_policies[pol][sw_idx, 0],
                    self.com_policies[pol][sw_idx, 1],
                    marker='o',
                    markersize=10,
                    color=colors[i % len(colors)],
                    label=f'{mapped_label} Switch Point' if sw_idx == switch_step else None
                )

        # --- Plot 2D End-Effector Trajectories (Contact Only) for EE0 ---
        b_idx, x_idx, y_idx, z_idx = (0, 1, 2, 3)  # EE0 indices
        bool_nom = self.cnt_nom[:, b_idx].astype(bool)
        x_nom = self.cnt_nom[:, x_idx]
        y_nom = self.cnt_nom[:, y_idx]
        x_nom_contact = x_nom[bool_nom]
        y_nom_contact = y_nom[bool_nom]

        ax.scatter(
            x_nom_contact,
            y_nom_contact,
            color='blue',
            label='EE0 MPC (contact only)',
            marker='o',
            s=10
        )

        for j, pol in enumerate(self.policies_to_compare):
            cnt_policy = self.cnt_policies[pol]
            bool_pol = cnt_policy[:, b_idx].astype(bool)
            x_pol = cnt_policy[:, x_idx]
            y_pol = cnt_policy[:, y_idx]
            x_pol_contact = x_pol[bool_pol]
            y_pol_contact = y_pol[bool_pol]

            mapped_label = get_policy_label(pol)
            ax.scatter(
                x_pol_contact,
                y_pol_contact,
                color=colors[j % len(colors)],
                label=f'EE0 {mapped_label} (contact only)',
                marker='o',
                s=10
            )

        # --- Adjust axis limits based on the full range of data ---
        all_x = np.concatenate([
            self.com_nom[:, 0],
            *(self.com_policies[pol][:, 0] for pol in self.policies_to_compare),
            self.cnt_nom[:, x_idx][self.cnt_nom[:, b_idx].astype(bool)]
        ])
        all_y = np.concatenate([
            self.com_nom[:, 1],
            *(self.com_policies[pol][:, 1] for pol in self.policies_to_compare),
            self.cnt_nom[:, y_idx][self.cnt_nom[:, b_idx].astype(bool)]
        ])

        # Remove NaNs
        all_x = all_x[~np.isnan(all_x)]
        all_y = all_y[~np.isnan(all_y)]

        ax.set_xlim(all_x.min() - 0.01, all_x.max() + 0.01)
        ax.set_ylim(all_y.min() - 0.01, all_y.max() + 0.01)

        # **Set equal aspect ratio**
        ax.set_aspect('equal', adjustable='box')

        # Formatting
        ax.set_xlabel("X-Position", fontsize=12)
        ax.set_ylabel("Y-Position", fontsize=12)
        ax.set_title("Combined 2D Trajectories (CoM + EE0)", fontsize=14)
        ax.legend(fontsize=10, loc='upper right')  # Position legend inside plot

        plt.tight_layout()

        # Save the figure with tight bounding box to include legend
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        plt.savefig(png_file, bbox_inches='tight')  # Ensure legend is included
        plt.show()

        print(f"Saved combined 2D trajectories plot to: {pkl_file}")
        print(f"Saved static image to: {png_file}")


    def plot_rmse_over_time(self, save_path=None):
        """
        Plots the instantaneous velocity error (magnitude) at each time step,
        comparing the commanded velocity (from comb) vs. the actual velocities
        (vx, vy) loaded from the .h5 file ('v' dataset).
        
        - You only need to pass `save_path` (the directory to save plots).
        - The figure is saved in both .pkl (interactive) and .png (static) formats.
        - Then displayed with plt.show().

        This method includes both policy trajectories and the nominal (MPC) trajectory.
        Additionally, it adds vertical lines to indicate when the controller switches
        from one velocity segment to another based on `comb`.
        """
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")
        if self.vel_nom is None or not self.vel_policies:
            raise ValueError("Velocity data not loaded. Call 'load_data()' first.")

        # Create the output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Parse comb & build commanded velocities
        # ------------------------------------------------------------------
        comb_list = ast.literal_eval(self.comb)  # e.g. [[0,1,0,0],[1,0.5,0.7,0]]
        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)  # e.g. if sim_time=2 => 2000 steps/segment
        total_length = segment_length * len(comb_list)

        commanded_vx = np.zeros(total_length)
        commanded_vy = np.zeros(total_length)

        switch_points = []  # To store indices where switches occur

        for i, seg in enumerate(comb_list):
            # According to your format: seg = [_, vx, vy, _]
            vx_cmd = seg[1]  # second entry
            vy_cmd = seg[2]  # third entry
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            commanded_vx[start_idx:end_idx] = vx_cmd
            commanded_vy[start_idx:end_idx] = vy_cmd

            # Record the switch point (except after the last segment)
            if i < len(comb_list) - 1:
                switch_points.append(end_idx)

        # ------------------------------------------------------------------
        # 2) Plot velocity error for each policy
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['red', 'green', 'orange']  # Colors for policies
        mpc_color = 'blue'  # Color for MPC

        # Plot Policy Trajectories
        for i, pol in enumerate(self.policies_to_compare):
            vel_data = self.vel_policies[pol]  # shape (N,2) => columns are (vx, vy)

            # The effective length of actual data might exceed or be less than total_length
            L = min(len(vel_data), total_length)
            if L <= 0:
                print(f"Skipping policy {pol} because velocity data is empty or no overlap.")
                continue

            # actual vx, vy
            actual_vx = vel_data[:L, 0]
            actual_vy = vel_data[:L, 1]

            # commanded vx, vy
            cmd_vx = commanded_vx[:L]
            cmd_vy = commanded_vy[:L]

            # velocity error = sqrt((vx_cmd - vx_actual)^2 + (vy_cmd - vy_actual)^2)
            error = np.sqrt((cmd_vx - actual_vx)**2 + (cmd_vy - actual_vy)**2)

            label_str = get_policy_label(pol) + " Velocity Error"
            ax.plot(error, label=label_str, color=colors[i % len(colors)], linewidth=1.5)

        # ------------------------------------------------------------------
        # 3) Compute and Plot Nominal (MPC) Velocity Error
        # ------------------------------------------------------------------
        # Extract nominal velocities up to total_length
        vel_nominal = self.vel_nom[:total_length, :]  # Shape (N, 2)

        # Commanded velocities aligned with nominal
        cmd_vx_nom = commanded_vx[:len(vel_nominal)]
        cmd_vy_nom = commanded_vy[:len(vel_nominal)]

        # Compute velocity error for nominal trajectory
        error_nom = np.sqrt((cmd_vx_nom - vel_nominal[:, 0])**2 + (cmd_vy_nom - vel_nominal[:, 1])**2)

        # Plot nominal velocity error
        label_nom = "MPC Velocity Error"
        ax.plot(error_nom, label=label_nom, color=mpc_color, linewidth=1.5, linestyle='--')

        # ------------------------------------------------------------------
        # 4) Add Vertical Lines for Segment Switches
        # ------------------------------------------------------------------
        for switch in switch_points:
            ax.axvline(x=switch, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
            # Optionally, add a label or annotation
            # ax.text(switch, ax.get_ylim()[1]*0.95, 'Switch', rotation=90, verticalalignment='top', color='grey')

        # ------------------------------------------------------------------
        # 5) Finishing touches (saving + show)
        # ------------------------------------------------------------------
        ax.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax.set_ylabel("Velocity Error (m/s)", fontsize=12)
        ax.set_title("Commanded vs. Actual Velocity Error Over Time", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(False)

        plt.tight_layout()

        # Create filenames
        pkl_file = os.path.join(policy_folder, f"RMSE_velocity_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"RMSE_velocity_over_time_{self.ID}.png")

        # Save interactive figure
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        # Save static image
        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

        # Show the figure
        plt.show()

# Example usage (keeping the user's original variables):
if __name__ == "__main__":
    from compareConf import policy, ep, comb, sim_time 
    folder = "/home/federico/biconmp_mujoco/project/results/tracking_perf_"
    analysis = TrajectoryAnalysis(folder, policy, comb, sim_time)
    analysis.load_data()
    save_dir = "/home/federico/biconmp_mujoco/project/results/plots"
    
    # Original 3D methods
    #analysis.plot_3d_trajectory(save_dir)
    #analysis.plot_ee_trajectories_3d_separate(save_dir)
    
    # New 2D projection methods
    #analysis.plot_2d_trajectory(save_dir)
    #analysis.plot_com_z_position(save_dir)
    #analysis.plot_ee_trajectories_2d_separate_contact_only(save_dir)
    #analysis.plot_combined_2d_trajectories(save_dir)
    analysis.plot_rmse_over_time(save_dir)
    #analysis.plot_rmse_over_time(save_dir, save_plot=True)
    #json_path = f"/home/federico/biconmp_mujoco/project/results/tracking_perf_{analysis.ID}_comparison_results.json"
    #analysis.plot_from_json_with_labels(json_path, os.path.join(save_dir, analysis.policy))
