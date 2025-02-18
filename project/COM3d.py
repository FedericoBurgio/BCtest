import ast
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkit
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
        self.vel_nom = None  # Nominal velocities (vx, vy, vz)

        # For multiple policies, we store data in dictionaries keyed by policy name
        self.com_policies = {}
        self.cnt_policies = {}
        self.vel_policies = {}  # Velocities for policies (vx, vy, vz)

    def load_data(self):
        """Loads the nominal and policy data from HDF5 files, padding shorter trajectories with the last value."""
        with h5py.File(self.file_nominal, 'r') as f_nom:
            states_nom = f_nom['states'][:]              # Existing CoM data
            cnt_nom = f_nom['cnt'][:] if 'cnt' in f_nom else None

            # --- Load velocities (assuming shape NxD, with D>=3, the first three are vx, vy, vz) ---
            if 'v' in f_nom:
                v_nom = f_nom['v'][:]
            else:
                raise ValueError(f"No 'v' dataset in nominal file {self.file_nominal}")

        # Store nominal CoM and contact points
        self.com_nom = states_nom[:, :3]  # unchanged
        self.cnt_nom = cnt_nom
        # Store nominal velocities (first three columns: vx, vy, vz)
        self.vel_nom = v_nom[:, 0:3]

        # Store the true length of the nominal trajectory
        self.true_nom_len = len(self.com_nom)

        # Find the maximum length across all policies
        max_length = self.true_nom_len

        # Prepare dicts to store policy data
        self.true_policy_lens = {}
        self.com_policies = {}
        self.cnt_policies = {}
        self.vel_policies = {}  # Holds velocities including vz

        # Load each policy data
        for pol in self.policies_to_compare:
            file_policy = f"{self.folder}{pol}{self.comb}{self.time}.h5"
            with h5py.File(file_policy, 'r') as f_policy:
                states_policy = f_policy['states'][:]
                cnt_policy = f_policy['cnt'][:] if 'cnt' in f_policy else None

                if 'v' in f_policy:
                    v_policy = f_policy['v'][:]
                else:
                    raise ValueError(f"No 'v' dataset in policy file {file_policy}")

            # Store true length
            self.true_policy_lens[pol] = len(states_policy)
            max_length = max(max_length, len(states_policy))

            # Store data
            self.com_policies[pol] = states_policy[:, :3]
            self.cnt_policies[pol] = cnt_policy
            self.vel_policies[pol] = v_policy[:, 0:3]  # store (vx, vy, vz)

        # Pad function
        def pad_with_last_value(data, target_length):
            if np.isnan(data).any():
                # If data contains NaNs, fill padded values with NaNs
                if data.ndim == 1:
                    padded = np.full(target_length, np.nan)
                    padded[:len(data)] = data
                else:
                    padded = np.full((target_length, data.shape[1]), np.nan)
                    padded[:len(data)] = data
            else:
                # Fill with last row if no NaNs
                padded = np.full((target_length, data.shape[1]), data[-1])
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

    def plot_z_rmse_over_time(self, save_path=None):
        """
        Plots the RMSE of Z-velocity over time for each policy compared to the nominal trajectory.
        - The RMSE is computed cumulatively up to each time step.
        - The plot is saved in both .pkl (interactive) and .png (static) formats.
        - Vertical lines indicate when the controller switches velocity segments based on the comb configuration.

        Parameters:
        - save_path (str): Directory to save the plots.
        """
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")
        if self.vel_nom is None or not self.vel_policies:
            raise ValueError("Velocity data not loaded. Call 'load_data()' first.")

        # Create the output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # Parse comb to build commanded velocities
        try:
            comb_list = ast.literal_eval(self.comb)  # e.g., [[0,1,0,0],[1,0.5,0.7,0]]
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)  # e.g., sim_time=1.3s => 1300 steps/segment
        total_length = segment_length * len(comb_list)

        # Limit to nominal's true length
        L = self.true_nom_len

        commanded_vz = np.zeros(L)

        switch_points = []  # Indices where switches occur within nominal's true length

        for i, seg in enumerate(comb_list):
            if len(seg) < 3:
                raise ValueError(f"Each comb segment must have at least 3 elements: {seg}")

            # Assuming seg = [_, _, vz, ...]
            vz_cmd = seg[2]
            start_idx = i * segment_length
            end_idx = start_idx + segment_length

            # Ensure indices do not exceed L
            if start_idx >= L:
                break
            end_idx = min(end_idx, L)

            commanded_vz[start_idx:end_idx] = vz_cmd

            if end_idx < L:
                switch_points.append(end_idx)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['red', 'green', 'orange']  # Colors for policies

        # Plot RMSE for each policy
        for i, pol in enumerate(self.policies_to_compare):
            vel_data = self.vel_policies[pol]  # shape (N,3) => columns are (vx, vy, vz)

            # Extract up to L
            vz_policy = vel_data[:L, 2]
            vz_nominal = self.vel_nom[:L, 2]

            # Compute squared error
            squared_error = (vz_nominal - vz_policy) ** 2

            # Compute cumulative sum of squared errors
            cumulative_squared_error = np.cumsum(squared_error)

            # Compute RMSE up to each time step
            time_steps = np.arange(1, L + 1)
            rmse_z = np.sqrt(cumulative_squared_error / time_steps)

            # Plot RMSE
            label_str = f"{get_policy_label(pol)} Z-Velocity RMSE"
            ax.plot(rmse_z, label=label_str, color=colors[i % len(colors)], linewidth=1.5)

        # # Plot RMSE for Nominal (MPC)
        # rmse_nom = np.zeros(L)
        # # Since nominal vs nominal should be zero, but for completeness, calculate it
        # squared_error_nom = (commanded_vz[:L] - self.vel_nom[:L, 2]) ** 2
        # cumulative_squared_error_nom = np.cumsum(squared_error_nom)
        # rmse_nom = np.sqrt(cumulative_squared_error_nom / np.arange(1, L + 1))
        # ax.plot(rmse_nom, label="MPC Z-Velocity RMSE", color='blue', linewidth=1.5, linestyle='--')

        # Add vertical lines for switch points
        for switch in switch_points:
            ax.axvline(x=switch, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        # Formatting
        ax.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax.set_ylabel("Z-Velocity RMSE (m/s)", fontsize=12)
        ax.set_title("Z-Velocity RMSE Over Time", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(False)

        plt.tight_layout()

        # Save the figure
        pkl_file = os.path.join(policy_folder, f"RMSE_z_velocity_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"RMSE_z_velocity_over_time_{self.ID}.png")

        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

        plt.show()

    def plot_combined_rmse_z_velocity(self, save_path=None):
        """
        Combines both the (vx, vy) velocity error and the Z-velocity RMSE into a single plot.
        This provides a comprehensive view of both horizontal and vertical velocity errors.
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
        try:
            comb_list = ast.literal_eval(self.comb)  # e.g., [[0,1,0,0],[1,0.5,0.7,0]]
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)  # e.g., sim_time=1.3s => 1300 steps/segment
        total_length = segment_length * len(comb_list)

        commanded_vx = np.zeros(total_length)
        commanded_vy = np.zeros(total_length)
        commanded_vz = np.zeros(total_length)

        switch_points = []  # To store indices where switches occur

        for i, seg in enumerate(comb_list):
            # According to your format: seg = [_, vx, vy, vz]
            vx_cmd = seg[1]  # second entry
            vy_cmd = seg[2]  # third entry
            vz_cmd = seg[3]  # fourth entry (if exists)
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            commanded_vx[start_idx:end_idx] = vx_cmd
            commanded_vy[start_idx:end_idx] = vy_cmd
            if len(seg) > 3:
                commanded_vz[start_idx:end_idx] = vz_cmd
            else:
                commanded_vz[start_idx:end_idx] = 0  # Default to 0 if not provided

            # Record the switch point (except after the last segment)
            if i < len(comb_list) - 1:
                switch_points.append(end_idx)

        # ------------------------------------------------------------------
        # 2) Compute RMSE for Z-velocity for each policy
        # ------------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()  # Create a second y-axis

        colors = ['red', 'green', 'orange']  # Colors for policies

        for i, pol in enumerate(self.policies_to_compare):
            vel_data = self.vel_policies[pol]  # shape (N,3) => columns are (vx, vy, vz)

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

            label_str = get_policy_label(pol) + " Velocity Error (vx, vy)"
            ax1.plot(error, label=label_str, color=colors[i % len(colors)], linewidth=1.5)

            # --- Plot Z-velocity RMSE ---
            vz_policy = vel_data[:L, 2]
            vz_nominal = self.vel_nom[:L, 2]

            # Compute squared error
            squared_error_z = (vz_nominal - vz_policy) ** 2

            # Compute cumulative sum of squared errors
            cumulative_squared_error_z = np.cumsum(squared_error_z)

            # Compute RMSE up to each time step
            time_steps = np.arange(1, L + 1)
            rmse_z = np.sqrt(cumulative_squared_error_z / time_steps)

            # Plot RMSE on the second y-axis
            label_rmse = get_policy_label(pol) + " Z-velocity RMSE"
            ax2.plot(rmse_z, label=label_rmse, color=colors[i % len(colors)], linewidth=1.5, linestyle='--')

        # Plot Nominal Trajectory (vx, vy)
        vel_nominal = self.vel_nom[:total_length, :]  # Shape (N, 3)

        # Commanded velocities aligned with nominal
        cmd_vx_nom = commanded_vx[:len(vel_nominal)]
        cmd_vy_nom = commanded_vy[:len(vel_nominal)]
        cmd_vz_nom = commanded_vz[:len(vel_nominal)]

        # Compute velocity error for nominal trajectory
        error_nom = np.sqrt((cmd_vx_nom - vel_nominal[:, 0])**2 + (cmd_vy_nom - vel_nominal[:, 1])**2)

        # Plot nominal velocity error
        label_nom = "MPC Velocity Error (vx, vy)"
        ax1.plot(error_nom, label=label_nom, color='blue', linewidth=1.5, linestyle='--')

        # --- Plot Z-velocity RMSE for Nominal ---
        error_nom_z = np.sqrt((cmd_vz_nom - vel_nominal[:, 2])**2)  # Since comparing to itself
        # Since it's comparing to itself, RMSE should be zero, but included for completeness
        rmse_nom_z = np.sqrt(np.cumsum((cmd_vz_nom - vel_nominal[:, 2])**2) / np.arange(1, len(vel_nominal) + 1))
        label_nom_rmse = "MPC Z-velocity RMSE"
        ax2.plot(rmse_nom_z, label=label_nom_rmse, color='blue', linewidth=1.5, linestyle=':')

        # ------------------------------------------------------------------
        # 3) Add Vertical Lines for Segment Switches
        # ------------------------------------------------------------------
        for switch in switch_points:
            ax1.axvline(x=switch, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        # ------------------------------------------------------------------
        # 4) Finalize Plot
        # ------------------------------------------------------------------
        ax1.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax1.set_ylabel("Velocity Error (m/s)", fontsize=12)
        ax2.set_ylabel("Z-Velocity RMSE (m/s)", fontsize=12)
        ax1.set_title("Velocity Error and Z-Velocity RMSE Over Time", fontsize=14)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

        ax1.grid(True)

        plt.tight_layout()

        # Create filenames
        pkl_file = os.path.join(policy_folder, f"Combined_RMSE_velocity_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"Combined_RMSE_velocity_over_time_{self.ID}.png")

        # Save interactive figure
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        # Save static image
        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

        # Show the figure
        plt.show()

    def plot_rmse_over_time(self, save_path=None):
        """
        Plots the instantaneous velocity error (magnitude) at each time step,
        comparing the commanded velocity (from comb) vs. the actual velocities
        (vx, vy) loaded from the .h5 file ('v' dataset).
        
        - You only need to pass save_path (the directory to save plots).
        - The figure is saved in both .pkl (interactive) and .png (static) formats.
        - Then displayed with plt.show().

        This method includes both policy trajectories and the nominal (MPC) trajectory.
        Additionally, it adds vertical lines to indicate when the controller switches
        from one velocity segment to another based on comb.
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
        try:
            comb_list = ast.literal_eval(self.comb)  # e.g., [[0,1,0,0],[1,0.5,0.7,0]]
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)  # e.g., sim_time=1.3s => 1300 steps/segment
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
            vel_data = self.vel_policies[pol]  # shape (N,3) => columns are (vx, vy, vz)

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
        vel_nominal = self.vel_nom[:total_length, :]  # Shape (N, 3)

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
        # 5) Finalize Plot
        # ------------------------------------------------------------------
        ax.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax.set_ylabel("Velocity Error (m/s)", fontsize=12)
        ax.set_title("Commanded vs. Actual Velocity Error Over Time - xy velocity RMSE", fontsize=14)
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

    def plot_rmse_over_time_combined(self, save_path=None):
        """
        Plots both the instantaneous velocity error (magnitude) for (vx, vy)
        and the cumulative RMSE for vz over time in a combined figure.

        - The plot includes both metrics for each policy.
        - The figure is saved in both .pkl (interactive) and .png (static) formats.
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
        try:
            comb_list = ast.literal_eval(self.comb)  # e.g., [[0,1,0,0],[1,0.5,0.7,0]]
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)  # e.g., sim_time=1.3s => 1300 steps/segment
        total_length = segment_length * len(comb_list)

        commanded_vx = np.zeros(total_length)
        commanded_vy = np.zeros(total_length)
        commanded_vz = np.zeros(total_length)

        switch_points = []  # To store indices where switches occur

        for i, seg in enumerate(comb_list):
            # According to your format: seg = [_, vx, vy, vz]
            vx_cmd = seg[1]  # second entry
            vy_cmd = seg[2]  # third entry
            vz_cmd = seg[3]  # fourth entry (if exists)
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            commanded_vx[start_idx:end_idx] = vx_cmd
            commanded_vy[start_idx:end_idx] = vy_cmd
            if len(seg) > 3:
                commanded_vz[start_idx:end_idx] = vz_cmd
            else:
                commanded_vz[start_idx:end_idx] = 0  # Default to 0 if not provided

            # Record the switch point (except after the last segment)
            if i < len(comb_list) - 1:
                switch_points.append(end_idx)

        # ------------------------------------------------------------------
        # 2) Compute RMSE for Z-velocity for each policy
        # ------------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()  # Create a second y-axis

        colors = ['red', 'green', 'orange']  # Colors for policies

        for i, pol in enumerate(self.policies_to_compare):
            vel_data = self.vel_policies[pol]  # shape (N,3) => columns are (vx, vy, vz)

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

            label_str = get_policy_label(pol) + " Velocity Error (vx, vy)"
            ax1.plot(error, label=label_str, color=colors[i % len(colors)], linewidth=1.5)

            # --- Plot Z-velocity RMSE ---
            vz_policy = vel_data[:L, 2]
            vz_nominal = self.vel_nom[:L, 2]

            # Compute squared error
            squared_error_z = (vz_nominal - vz_policy) ** 2

            # Compute cumulative sum of squared errors
            cumulative_squared_error_z = np.cumsum(squared_error_z)

            # Compute RMSE up to each time step
            time_steps = np.arange(1, L + 1)
            rmse_z = np.sqrt(cumulative_squared_error_z / time_steps)

            # Plot RMSE on the second y-axis
            label_rmse = get_policy_label(pol) + " Z-velocity RMSE"
            ax2.plot(rmse_z, label=label_rmse, color=colors[i % len(colors)], linewidth=1.5, linestyle='--')

        # Plot Nominal Trajectory (vx, vy)
        vel_nominal = self.vel_nom[:total_length, :]  # Shape (N, 3)

        # Commanded velocities aligned with nominal
        cmd_vx_nom = commanded_vx[:len(vel_nominal)]
        cmd_vy_nom = commanded_vy[:len(vel_nominal)]
        cmd_vz_nom = commanded_vz[:len(vel_nominal)]

        # Compute velocity error for nominal trajectory
        error_nom = np.sqrt((cmd_vx_nom - vel_nominal[:, 0])**2 + (cmd_vy_nom - vel_nominal[:, 1])**2)

        # Plot nominal velocity error
        label_nom = "MPC Velocity Error (vx, vy)"
        ax1.plot(error_nom, label=label_nom, color='blue', linewidth=1.5, linestyle='--')

        # --- Plot Z-velocity RMSE for Nominal ---
        error_nom_z = np.sqrt((cmd_vz_nom - vel_nominal[:, 2])**2)  # Since comparing to itself
        # Since it's comparing to itself, RMSE should be zero, but included for completeness
        rmse_nom_z = np.sqrt(np.cumsum((cmd_vz_nom - vel_nominal[:, 2])**2) / np.arange(1, len(vel_nominal) + 1))
        label_nom_rmse = "MPC Z-velocity RMSE"
        ax2.plot(rmse_nom_z, label=label_nom_rmse, color='blue', linewidth=1.5, linestyle=':')

        # ------------------------------------------------------------------
        # 3) Add Vertical Lines for Segment Switches
        # ------------------------------------------------------------------
        for switch in switch_points:
            ax1.axvline(x=switch, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        # ------------------------------------------------------------------
        # 4) Finalize Plot
        # ------------------------------------------------------------------
        ax1.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax1.set_ylabel("Velocity Error (m/s)", fontsize=12)
        ax2.set_ylabel("Z-Velocity RMSE (m/s)", fontsize=12)
        ax1.set_title("Velocity Error and Z-Velocity RMSE Over Time", fontsize=14)

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

        ax1.grid(True)

        plt.tight_layout()

        # Create filenames
        pkl_file = os.path.join(policy_folder, f"Combined_RMSE_velocity_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"Combined_RMSE_velocity_over_time_{self.ID}.png")

        # Save interactive figure
        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        # Save static image
        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

        # Show the figure
        plt.show()
        
    def plot_mae_over_time_xy(self, save_path=None):
        """
        Plots the *cumulative* Mean Absolute Error (MAE) over time (separately for x and y),
        comparing the commanded velocity (from comb) vs. the actual velocities (vx, vy)
        from both the nominal (MPC) and the policy files.

        - Only plots data up to the "true" trajectory length for each (nominal/policy).
        - Vertical dashed lines indicate switch points, truncated to each trajectory's length.
        - Saves figure in both .pkl (interactive) and .png (static) formats.
        """
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")
        if self.vel_nom is None or not self.vel_policies:
            raise ValueError("Velocity data not loaded. Call 'load_data()' first.")

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # 1) Parse comb to build commanded velocities in x, y
        try:
            comb_list = ast.literal_eval(self.comb)  # e.g., [[0, vx, vy, ...], [1, vx, vy, ...]]
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)
        total_length = segment_length * len(comb_list)

        commanded_vx = np.zeros(total_length)
        commanded_vy = np.zeros(total_length)

        switch_points = []
        for i, seg in enumerate(comb_list):
            vx_cmd = seg[1]  # e.g. seg = [_, vx, vy, ...]
            vy_cmd = seg[2]
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            commanded_vx[start_idx:end_idx] = vx_cmd
            commanded_vy[start_idx:end_idx] = vy_cmd
            if i < len(comb_list) - 1:
                switch_points.append(end_idx)

        # 2) Prepare figure with 2 subplots: top for X error, bottom for Y error
        fig, (ax_x, ax_y) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=100, sharex=True)

        # Colors for different policies
        colors = ['red', 'green', 'orange']
        color_nominal = 'blue'

        # 3) Handle nominal (MPC) velocity => only up to true_nom_len
        L_nom = self.true_nom_len
        vel_nominal = self.vel_nom[:L_nom, :]  # shape (L_nom, 3)
        cmd_vx_nom = commanded_vx[:L_nom]
        cmd_vy_nom = commanded_vy[:L_nom]

        abs_err_x_nom = np.abs(cmd_vx_nom - vel_nominal[:, 0])
        abs_err_y_nom = np.abs(cmd_vy_nom - vel_nominal[:, 1])

        cmae_x_nom = np.cumsum(abs_err_x_nom) / np.arange(1, L_nom + 1)
        cmae_y_nom = np.cumsum(abs_err_y_nom) / np.arange(1, L_nom + 1)

        ax_x.plot(
            range(L_nom), cmae_x_nom,
            label="MPC MAE (x)",
            color=color_nominal,
            linestyle='--'
        )
        ax_y.plot(
            range(L_nom), cmae_y_nom,
            label="MPC MAE (y)",
            color=color_nominal,
            linestyle='--'
        )

        # 4) Handle each policy
        for i, pol in enumerate(self.policies_to_compare):
            L_pol = self.true_policy_lens[pol]
            if L_pol <= 0:
                continue

            vel_data = self.vel_policies[pol][:L_pol]  # shape (L_pol, 3)
            cmd_vx_pol = commanded_vx[:L_pol]
            cmd_vy_pol = commanded_vy[:L_pol]

            actual_vx = vel_data[:, 0]
            actual_vy = vel_data[:, 1]

            abs_err_x = np.abs(cmd_vx_pol - actual_vx)
            abs_err_y = np.abs(cmd_vy_pol - actual_vy)

            cmae_x = np.cumsum(abs_err_x) / np.arange(1, L_pol + 1)
            cmae_y = np.cumsum(abs_err_y) / np.arange(1, L_pol + 1)

            ax_x.plot(
                range(L_pol), cmae_x,
                label=f"{pol} MAE (x)",
                color=colors[i % len(colors)]
            )
            ax_y.plot(
                range(L_pol), cmae_y,
                label=f"{pol} MAE (y)",
                color=colors[i % len(colors)]
            )

        # 5) Add vertical dashed lines for switch points, limited by each axis's max
        max_len = max(L_nom, max(self.true_policy_lens.values()))
        for sp in switch_points:
            if sp < max_len:
                ax_x.axvline(x=sp, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_y.axvline(x=sp, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        # 6) Formatting
        ax_x.set_ylabel("MAE in X (m/s)", fontsize=12)
        ax_y.set_ylabel("MAE in Y (m/s)", fontsize=12)
        ax_y.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax_x.set_title("Cumulative Mean Absolute Error (X, Y) Over Time", fontsize=14)

        ax_x.legend(fontsize=9)
        ax_y.legend(fontsize=9)

        plt.tight_layout()

        # 7) Save the figure
        pkl_file = os.path.join(policy_folder, f"MAE_xy_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"MAE_xy_over_time_{self.ID}.png")

        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive MAE figure to: {pkl_file}")

        plt.savefig(png_file)
        print(f"Saved static MAE image to: {png_file}")

        plt.show()

    def plot_commanded_vs_actual_velocity_xy(self, save_path=None):
        """
        Plots commanded vs. actual velocities (vx, vy) over time in two subplots:
          - Top Subplot: vx (commanded + actual for each policy + nominal)
          - Bottom Subplot: vy (commanded + actual for each policy + nominal)

        - Only plots data up to the "true" trajectory length for each.
        - Includes vertical dashed lines at switch points (within that length).
        - Saves the figure as both .pkl and .png, then displays it.
        """
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")
        if self.vel_nom is None or not self.vel_policies:
            raise ValueError("Velocity data not loaded. Call 'load_data()' first.")

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # 1) Parse comb to build commanded velocities in x, y
        try:
            comb_list = ast.literal_eval(self.comb)
        except Exception as e:
            raise ValueError(f"Error parsing comb: {e}")

        sim_t = float(self.time)
        segment_length = int(sim_t * 1000)
        total_length = segment_length * len(comb_list)

        commanded_vx = np.zeros(total_length)
        commanded_vy = np.zeros(total_length)

        switch_points = []
        for i, seg in enumerate(comb_list):
            vx_cmd = seg[1]
            vy_cmd = seg[2]
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            commanded_vx[start_idx:end_idx] = vx_cmd
            commanded_vy[start_idx:end_idx] = vy_cmd

            if i < len(comb_list) - 1:
                switch_points.append(end_idx)

        # 2) Create figure with 2 subplots
        fig, (ax_vx, ax_vy) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=100, sharex=True)

        # We'll track the max length so we know how far the x-axis should go
        max_len = max(self.true_nom_len, max(self.true_policy_lens.values()))

        # 3) Plot commanded velocities (up to max_len for visual consistency)
        time_steps = np.arange(max_len)
        ax_vx.plot(
            time_steps,
            commanded_vx[:max_len],
            label="Commanded vx",
            color='black',
            linestyle='--'
        )
        ax_vy.plot(
            time_steps,
            commanded_vy[:max_len],
            label="Commanded vy",
            color='black',
            linestyle='--'
        )

        # 4) Plot Nominal (MPC) velocities, up to self.true_nom_len
        L_nom = self.true_nom_len
        vel_nominal = self.vel_nom[:L_nom]
        ax_vx.plot(
            np.arange(L_nom),
            vel_nominal[:, 0],
            label="MPC actual vx",
            color='blue'
        )
        ax_vy.plot(
            np.arange(L_nom),
            vel_nominal[:, 1],
            label="MPC actual vy",
            color='blue'
        )

        # 5) Plot each policy's actual velocities up to their true length
        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            L_pol = self.true_policy_lens[pol]
            if L_pol <= 0:
                continue

            vel_data = self.vel_policies[pol][:L_pol]
            actual_vx = vel_data[:, 0]
            actual_vy = vel_data[:, 1]

            label_vx = f"{pol} actual vx"
            label_vy = f"{pol} actual vy"
            ax_vx.plot(
                np.arange(L_pol),
                actual_vx,
                label=label_vx,
                color=colors[i % len(colors)]
            )
            ax_vy.plot(
                np.arange(L_pol),
                actual_vy,
                label=label_vy,
                color=colors[i % len(colors)]
            )

        # 6) Add vertical switch lines for points < max_len
        for sp in switch_points:
            if sp < max_len:
                ax_vx.axvline(x=sp, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
                ax_vy.axvline(x=sp, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

        # 7) Labels and legends
        ax_vx.set_ylabel("vx (m/s)", fontsize=12)
        ax_vy.set_ylabel("vy (m/s)", fontsize=12)
        ax_vy.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax_vx.set_title("Commanded vs. Actual Velocities (X and Y)", fontsize=14)

        ax_vx.legend(fontsize=9, loc='best')
        ax_vy.legend(fontsize=9, loc='best')

        plt.tight_layout()

        # 8) Save
        pkl_file = os.path.join(policy_folder, f"commanded_vs_actual_vxy_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"commanded_vs_actual_vxy_{self.ID}.png")

        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

        plt.show()
    def plot_z_mae_over_time(self, save_path=None):
        """
        Plots the *cumulative mean absolute error* (MAE) of Z-velocity over time for each policy 
        compared to the *nominal* Z-velocity (from MPC). The error is computed as the absolute difference 
        between the policy's Z-velocity and the nominal's Z-velocity.
        
        Only the true (unpadded) portions of the trajectories are plotted. For each policy the valid data 
        is plotted over its true length and the remainder (up to the nominal’s true length) is filled with NaN 
        values so that no line is drawn beyond the unpadded data.
        
        - Saves the figure as both a .pkl (interactive) and a .png (static) file, then displays it.
        """
        if save_path is None:
            raise ValueError("save_path must be specified to save results.")
        if self.vel_nom is None or not self.vel_policies:
            raise ValueError("Velocity data not loaded. Call 'load_data()' first.")

        # Create output folder if needed
        policy_folder = os.path.join(save_path, self.policy)
        os.makedirs(policy_folder, exist_ok=True)

        # ----------------------------------------------------------------------
        # 1) Determine the true (unpadded) trajectory length for the nominal data.
        # ----------------------------------------------------------------------
        L_nom = self.true_nom_len
        if L_nom <= 0:
            raise ValueError("Nominal trajectory length is invalid.")

        # ----------------------------------------------------------------------
        # 2) Create figure and set x-axis limit to the nominal trajectory's true length.
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_xlim([0, L_nom])

        # ----------------------------------------------------------------------
        # 3) (Optional) Nominal (MPC) baseline.
        # ----------------------------------------------------------------------
        # (The nominal trajectory compared to itself would yield zero error.)
        # You could plot a baseline at zero if desired; here it is omitted so that only policy errors are shown.
        # cmae_z_nom = np.zeros(L_nom)
        # ax.plot(range(L_nom), cmae_z_nom, label="MPC Z-MAE", color='blue', linewidth=2, linestyle='--')

        # ----------------------------------------------------------------------
        # 4) For each policy: trim the data to its true (unpadded) length, compute CMAE in Z,
        #    then pad with NaNs so nothing is drawn after the true data.
        # ----------------------------------------------------------------------
        colors = ['red', 'green', 'orange']
        for i, pol in enumerate(self.policies_to_compare):
            # Get the true (unpadded) length for this policy.
            L_pol = self.true_policy_lens[pol]
            if L_pol <= 0:
                continue

            # Use the smaller length between the policy and the nominal trajectory.
            L_pol = min(L_pol, L_nom)

            # Extract the valid (unpadded) portion of the z velocities.
            policy_z = self.vel_policies[pol][:L_pol, 2]
            nominal_z = self.vel_nom[:L_pol, 2]

            # Compute the absolute error and its cumulative mean.
            abs_err_pol = np.abs(policy_z - nominal_z)
            cmae_z_pol = np.cumsum(abs_err_pol) / np.arange(1, L_pol + 1)

            # Create an array of length L_nom, fill valid indices with computed data, and pad the rest with NaN.
            plot_data = np.full(L_nom, np.nan)
            plot_data[:L_pol] = cmae_z_pol

            ax.plot(
                range(L_nom), plot_data,
                label=f"{pol} Z-MAE",
                color=colors[i % len(colors)],
                linewidth=2
            )

        # ----------------------------------------------------------------------
        # 5) Formatting the plot.
        # ----------------------------------------------------------------------
        ax.set_xlabel("Time Step (each = 1 ms)", fontsize=12)
        ax.set_ylabel("Z-Velocity MAE (m/s)", fontsize=12)
        ax.set_title("Cumulative Mean Absolute Error in Z Velocity Over Time\n(Policy vs Nominal)", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True)
        plt.tight_layout()

        # ----------------------------------------------------------------------
        # 6) Save the figure.
        # ----------------------------------------------------------------------
        pkl_file = os.path.join(policy_folder, f"MAE_z_velocity_over_time_{self.ID}.pkl")
        png_file = os.path.join(policy_folder, f"MAE_z_velocity_over_time_{self.ID}.png")

        with open(pkl_file, 'wb') as f:
            pickle.dump(fig, f)
        print(f"Saved interactive figure to: {pkl_file}")

        plt.savefig(png_file)
        print(f"Saved static image to: {png_file}")

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
    analysis.plot_2d_trajectory(save_dir)
    analysis.plot_com_z_position(save_dir)
    #analysis.plot_ee_trajectories_2d_separate_contact_only(save_dir)
    #analysis.plot_combined_2d_trajectories(save_dir)
    
    # New Z RMSE plot
    # analysis.plot_rmse_over_time(save_dir)
    # analysis.plot_z_rmse_over_time(save_dir)
    # analysis.plot_combined_rmse_z_velocity(save_dir)
    #analysis.plot_mae_over_time_xy(save_dir)
    analysis.plot_commanded_vs_actual_velocity_xy(save_dir) 
    analysis.plot_z_mae_over_time(save_dir) 
    # Existing RMSE plot
    
    # Combined RMSE and Z RMSE plot
    #analysis.plot_rmse_over_time_combined(save_dir)
    
    #analysis.plot_rmse_over_time(save_dir, save_plot=True)
    #json_path = f"/home/federico/biconmp_mujoco/project/results/tracking_perf_{analysis.ID}_comparison_results.json"
    #analysis.plot_from_json_with_labels(json_path, os.path.join(save_dir, analysis.policy))
