import os

def plot_from_json_with_labels(json_path, save_dir):
    """
    Loads JSON data and generates labeled scatter plots for different variable categories.

    Parameters:
    - json_path: Path to the JSON file.
    - save_dir: Directory to save the generated plots.
    """
    import json
    import matplotlib.pyplot as plt

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    # Extract relevant data
    rmse_states = data.get("rmse_states_per_variable", [])
    total_rmse_states = data.get("total_rmse_states", 0)
    rmse_velocities = data.get("rmse_velocities_per_variable", [])
    total_rmse_velocities = data.get("total_rmse_velocities", 0)

    # Define labels for different variable categories
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
    plt.savefig(os.path.join(save_dir, ("base_position_rmse_" + ID + ".png")))
    plt.close()

    # Plot quaternion RMSE
    plt.figure()
    plt.scatter(range(len(quaternion_states)), quaternion_states, label="Quaternion RMSE")
    plt.xticks(range(len(quaternion_states)), ["qx", "qy", "qz", "qw"])
    plt.ylabel("RMSE")
    plt.title("Quaternion RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_dir, ("quaternion_rmse_" + ID + ".png")))
    plt.close()

    # Plot joint position RMSE
    plt.figure()
    plt.scatter(range(len(joint_states)), joint_states, label="Joint Position RMSE")
    plt.xticks(range(len(joint_states)), [f"q{i+1}" for i in range(len(joint_states))], rotation=90)
    plt.ylabel("RMSE")
    plt.title("Joint Position RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_dir, ("joint_position_rmse_" + ID + ".png")))
    plt.close()

    # Plot base linear velocity RMSE
    plt.figure()
    plt.scatter(range(len(base_velocities)), base_velocities, label="Base Linear Velocity RMSE")
    plt.xticks(range(len(base_velocities)), ["vx", "vy", "vz"])
    plt.ylabel("RMSE")
    plt.title("Base Linear Velocity RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_dir, ("base_linear_velocity_rmse_" + ID + ".png")))
    plt.close()

    # Plot angular velocity RMSE
    plt.figure()
    plt.scatter(range(len(angular_velocities)), angular_velocities, label="Angular Velocity RMSE")
    plt.xticks(range(len(angular_velocities)), ["wx", "wy", "wz"])
    plt.ylabel("RMSE")
    plt.title("Angular Velocity RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_dir, ("angular_velocity_rmse_" + ID + ".png")))
    plt.close()

    # Plot joint velocity RMSE
    plt.figure()
    plt.scatter(range(len(joint_velocities)), joint_velocities, label="Joint Velocity RMSE")
    plt.xticks(range(len(joint_velocities)), [f"v{i+1}" for i in range(len(joint_velocities))], rotation=90)
    plt.ylabel("RMSE")
    plt.title("Joint Velocity RMSE")
    plt.legend()
    plt.savefig(os.path.join(save_dir, ("joint_velocity_rmse_" + ID + ".png")))
    plt.close()

    print(f"Plots saved in {save_dir}")


from compareConf import policy, ep, comb, sim_time 
comb = str(comb)
sim_time = str(sim_time)
ID =  policy + comb + sim_time

json_path = "/home/federico/biconmp_mujoco/project/results/tracking_perf_" + ID + "_comparison_results.json"
save_dir = "/home/federico/biconmp_mujoco/project/results/plots/" + policy
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#id = json_path.replace("/home/federico/biconmp_mujoco/project/results/tracking_perf_Nom", "")
#ID = id.replace("_comparison_results.json", "")
#del id

# Run the plotting function
plot_from_json_with_labels(json_path, save_dir)
