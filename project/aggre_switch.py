import os
import json
import random
import itertools
import numpy as np

class SequenceResultsAggregator:
    def __init__(self, comb, sim_time_list, policies, results_folder):
        """
        Initialize the aggregator.

        :param comb: List of episodes, each containing 4 sub-combos.
        :param sim_time_list: List of simulation times corresponding to each episode.
        :param policies: List of policies to aggregate (e.g., ["Nom", "081059", ...]).
        :param results_folder: Directory where JSON result files are stored.
        """
        if len(comb) != len(sim_time_list):
            raise ValueError("Length of comb and sim_time_list must match.")

        self.comb = comb
        self.sim_time_list = sim_time_list
        self.policies = policies
        self.results_folder = results_folder

        # ---------------- NOMINAL (MPC) ACCUMULATORS ----------------
        # RMSE
        self.nominal_rmse_vxy_list = []
        self.nominal_rmse_vx_list = []
        self.nominal_rmse_vy_list = []
        # MAE
        self.nominal_mae_vxy_list = []
        self.nominal_mae_vx_list = []
        self.nominal_mae_vy_list = []
        # Cumulative MAE
        self.nominal_cum_mae_vxy_list = []
        self.nominal_cum_mae_vx_list = []
        self.nominal_cum_mae_vy_list = []

        # Survived steps
        self.nominal_survived_steps_sum = 0
        self.nominal_total_steps_sum = 0

        # ---------------- LEARNED-POLICY ACCUMULATORS ----------------
        # Each policy has its own set of lists to accumulate data
        self.policy_stats = {}
        for pol in policies:
            if pol != "Nom":
                self.policy_stats[pol] = {
                    # RMSE
                    "rmse_vxy_list": [],
                    "rmse_vx_list": [],
                    "rmse_vy_list": [],
                    # MAE
                    "mae_vxy_list": [],
                    "mae_vx_list": [],
                    "mae_vy_list": [],
                    # Cumulative MAE
                    "cum_mae_vxy_list": [],
                    "cum_mae_vx_list": [],
                    "cum_mae_vy_list": [],
                    # z-vel metrics
                    "z_rmse_list": [],
                    "z_mae_list": [],
                    # Survived steps
                    "survived_steps_sum": 0,
                    "total_steps_sum": 0
                }

        # Per-episode metrics to store for final summary
        self.per_episode_metrics = []

    def aggregate(self):
        """
        Aggregate metrics from JSON files for all episodes and policies.
        """
        for i, episode_subcombos in enumerate(self.comb):
            st = self.sim_time_list[i]
            total_segments = len(episode_subcombos)  # Should be 4
            total_steps_for_episode = int(st * 1000 * total_segments)

            # You might have your own logic for naming,
            # but here we just convert the subcombo list to a string:
            combo_str = str(episode_subcombos)

            # ---------------- Initialize per-episode data ----------------
            episode_data = {
                "episode_index": i,
                "comb": episode_subcombos,
                "sim_time": st,
                "Nominal": {},
                "Policies": {}
            }

            # ---------------- A) NOMINAL (MPC) ----------------
            if "Nom" in self.policies:
                # NOTE: Use "tracking_perf_nominal" prefix,
                # and look for "nominal_vs_commanded" in the JSON
                nominal_prefix = "tracking_perf_Nom"
                nominal_suffix = "_cmd_vel_comparison.json"
                json_name = f"{nominal_prefix}{combo_str}{st}{nominal_suffix}"
                json_path = os.path.join(self.results_folder, json_name)

                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    # LOOK FOR "nominal_vs_commanded" INSTEAD OF "mpc_vs_commanded"
                    if "mpc_vs_commanded" in data:
                        sub_data = data["mpc_vs_commanded"]

                        # -- RMSE metrics --
                        rmse_vxy = sub_data.get("rmse_total_vxy", np.nan)
                        rmse_vx = sub_data.get("rmse_vx", np.nan)
                        rmse_vy = sub_data.get("rmse_vy", np.nan)

                        # -- MAE metrics --
                        mae_vxy = sub_data.get("mae_total_vxy", np.nan)
                        mae_vx = sub_data.get("mae_vx", np.nan)
                        mae_vy = sub_data.get("mae_vy", np.nan)

                        # -- Cumulative MAE metrics --
                        cum_mae_vxy = sub_data.get("cumulative_mae_total_vxy", np.nan)
                        cum_mae_vx = sub_data.get("cumulative_mae_vx", np.nan)
                        cum_mae_vy = sub_data.get("cumulative_mae_vy", np.nan)

                        # -- Survived timesteps --
                        survived_steps = sub_data.get("survived_timesteps", 0)

                        # Accumulate Nominal metrics
                        # RMSE
                        self.nominal_rmse_vxy_list.append(rmse_vxy)
                        self.nominal_rmse_vx_list.append(rmse_vx)
                        self.nominal_rmse_vy_list.append(rmse_vy)
                        # MAE
                        self.nominal_mae_vxy_list.append(mae_vxy)
                        self.nominal_mae_vx_list.append(mae_vx)
                        self.nominal_mae_vy_list.append(mae_vy)
                        # Cumulative MAE
                        self.nominal_cum_mae_vxy_list.append(cum_mae_vxy)
                        self.nominal_cum_mae_vx_list.append(cum_mae_vx)
                        self.nominal_cum_mae_vy_list.append(cum_mae_vy)

                        # Survivability
                        self.nominal_survived_steps_sum += survived_steps
                        self.nominal_total_steps_sum += total_steps_for_episode

                        # Store per-episode Nominal data
                        episode_data["Nominal"] = {
                            "rmse_total_vxy": rmse_vxy,
                            "rmse_vx": rmse_vx,
                            "rmse_vy": rmse_vy,
                            "mae_total_vxy": mae_vxy,
                            "mae_vx": mae_vx,
                            "mae_vy": mae_vy,
                            "cumulative_mae_total_vxy": cum_mae_vxy,
                            "cumulative_mae_vx": cum_mae_vx,
                            "cumulative_mae_vy": cum_mae_vy,
                            "survivability_percent": (
                                (survived_steps / total_steps_for_episode) * 100.0
                                if total_steps_for_episode > 0 else 0.0
                            )
                        }
                    else:
                        print(f"[Warning] 'nominal_vs_commanded' key not found in: {json_path}")
                else:
                    print(f"[Warning] Nominal JSON not found: {json_path}")

            # ---------------- B) LEARNED POLICIES ----------------
            for pol in self.policies:
                if pol == "Nom":
                    continue
                pol_prefix = f"tracking_perf_{pol}"
                pol_suffix = "_cmd_vel_comparison.json"
                pol_json_name = f"{pol_prefix}{combo_str}{st}{pol_suffix}"
                pol_json_path = os.path.join(self.results_folder, pol_json_name)
                
                if os.path.isfile(pol_json_path):
                    with open(pol_json_path, "r") as f:
                        data = json.load(f)

                    # The relevant policy block is "policy_vs_commanded"
                    if "policy_vs_commanded" in data:
                        sub_data = data["policy_vs_commanded"]

                        # -- RMSE metrics --
                        rmse_vxy = sub_data.get("rmse_total_vxy", np.nan)
                        rmse_vx = sub_data.get("rmse_vx", np.nan)
                        rmse_vy = sub_data.get("rmse_vy", np.nan)

                        # -- MAE metrics --
                        mae_vxy = sub_data.get("mae_total_vxy", np.nan)
                        mae_vx = sub_data.get("mae_vx", np.nan)
                        mae_vy = sub_data.get("mae_vy", np.nan)

                        # -- Cumulative MAE metrics --
                        cum_mae_vxy = sub_data.get("cumulative_mae_total_vxy", np.nan)
                        cum_mae_vx = sub_data.get("cumulative_mae_vx", np.nan)
                        cum_mae_vy = sub_data.get("cumulative_mae_vy", np.nan)

                        # -- Survived timesteps --
                        survived_steps = sub_data.get("survived_timesteps", 0)

                        # -- z-vel metrics if present --
                        z_rmse_val = data.get("z_vel_rmse_mpc_vs_policy", np.nan)
                        z_mae_val = data.get("z_vel_mae_mpc_vs_policy", np.nan)

                        # Accumulate Policy metrics
                        self.policy_stats[pol]["rmse_vxy_list"].append(rmse_vxy)
                        self.policy_stats[pol]["rmse_vx_list"].append(rmse_vx)
                        self.policy_stats[pol]["rmse_vy_list"].append(rmse_vy)

                        self.policy_stats[pol]["mae_vxy_list"].append(mae_vxy)
                        self.policy_stats[pol]["mae_vx_list"].append(mae_vx)
                        self.policy_stats[pol]["mae_vy_list"].append(mae_vy)

                        self.policy_stats[pol]["cum_mae_vxy_list"].append(cum_mae_vxy)
                        self.policy_stats[pol]["cum_mae_vx_list"].append(cum_mae_vx)
                        self.policy_stats[pol]["cum_mae_vy_list"].append(cum_mae_vy)

                        if not np.isnan(z_rmse_val):
                            self.policy_stats[pol]["z_rmse_list"].append(z_rmse_val)
                        if not np.isnan(z_mae_val):
                            self.policy_stats[pol]["z_mae_list"].append(z_mae_val)

                        self.policy_stats[pol]["survived_steps_sum"] += survived_steps
                        self.policy_stats[pol]["total_steps_sum"] += total_steps_for_episode

                        # Store per-episode Policy data
                        episode_data["Policies"][pol] = {
                            "rmse_total_vxy": rmse_vxy,
                            "rmse_vx": rmse_vx,
                            "rmse_vy": rmse_vy,
                            "mae_total_vxy": mae_vxy,
                            "mae_vx": mae_vx,
                            "mae_vy": mae_vy,
                            "cumulative_mae_total_vxy": cum_mae_vxy,
                            "cumulative_mae_vx": cum_mae_vx,
                            "cumulative_mae_vy": cum_mae_vy,
                            "z_vel_rmse_mpc_vs_policy": z_rmse_val,
                            "z_vel_mae_mpc_vs_policy": z_mae_val,
                            "survivability_percent": (
                                (survived_steps / total_steps_for_episode) * 100.0
                                if total_steps_for_episode > 0 else 0.0
                            )
                        }
                    else:
                        print(f"[Warning] 'policy_vs_commanded' key not found in: {pol_json_path}")
                else:
                    print(f"[Warning] Policy {pol} JSON not found: {pol_json_path}")

            # Append per-episode data
            self.per_episode_metrics.append(episode_data)

    def save_summary(self, output_json_path):
        """
        Save the aggregated summary to a JSON file.
        """
        seq_summary = {
            "num_episodes": len(self.comb),
            "episodes": self.per_episode_metrics
        }

        # ---------------- NOMINAL (MPC) SUMMARY ----------------
        if "Nom" in self.policies:
            # RMSE
            mean_rmse_nom = (float(np.mean(self.nominal_rmse_vxy_list))
                             if self.nominal_rmse_vxy_list else None)
            mean_rmse_vx_nom = (float(np.mean(self.nominal_rmse_vx_list))
                                if self.nominal_rmse_vx_list else None)
            mean_rmse_vy_nom = (float(np.mean(self.nominal_rmse_vy_list))
                                if self.nominal_rmse_vy_list else None)

            # MAE
            mean_mae_nom = (float(np.mean(self.nominal_mae_vxy_list))
                            if self.nominal_mae_vxy_list else None)
            mean_mae_vx_nom = (float(np.mean(self.nominal_mae_vx_list))
                               if self.nominal_mae_vx_list else None)
            mean_mae_vy_nom = (float(np.mean(self.nominal_mae_vy_list))
                               if self.nominal_mae_vy_list else None)

            # Cumulative MAE
            mean_cum_mae_nom = (float(np.mean(self.nominal_cum_mae_vxy_list))
                                if self.nominal_cum_mae_vxy_list else None)
            mean_cum_mae_vx_nom = (float(np.mean(self.nominal_cum_mae_vx_list))
                                   if self.nominal_cum_mae_vx_list else None)
            mean_cum_mae_vy_nom = (float(np.mean(self.nominal_cum_mae_vy_list))
                                   if self.nominal_cum_mae_vy_list else None)

            # Survivability
            survival_nom = (
                (self.nominal_survived_steps_sum / self.nominal_total_steps_sum) * 100.0
                if self.nominal_total_steps_sum > 0 else 0.0
            )

            seq_summary["Nominal"] = {
                # RMSE
                "mean_rmse_total_vxy": mean_rmse_nom,
                "mean_rmse_vx": mean_rmse_vx_nom,
                "mean_rmse_vy": mean_rmse_vy_nom,
                # MAE
                "mean_mae_total_vxy": mean_mae_nom,
                "mean_mae_vx": mean_mae_vx_nom,
                "mean_mae_vy": mean_mae_vy_nom,
                # Cumulative MAE
                "mean_cumulative_mae_total_vxy": mean_cum_mae_nom,
                "mean_cumulative_mae_vx": mean_cum_mae_vx_nom,
                "mean_cumulative_mae_vy": mean_cum_mae_vy_nom,
                # Survivability
                "survivability_percent": survival_nom
            }

        # ---------------- LEARNED POLICIES SUMMARY ----------------
        pol_data = {}
        for pol in self.policies:
            if pol == "Nom":
                continue

            stats = self.policy_stats[pol]

            # RMSE
            mean_rmse_vxy = (float(np.mean(stats["rmse_vxy_list"]))
                             if stats["rmse_vxy_list"] else None)
            mean_rmse_vx = (float(np.mean(stats["rmse_vx_list"]))
                            if stats["rmse_vx_list"] else None)
            mean_rmse_vy = (float(np.mean(stats["rmse_vy_list"]))
                            if stats["rmse_vy_list"] else None)

            # MAE
            mean_mae_vxy = (float(np.mean(stats["mae_vxy_list"]))
                            if stats["mae_vxy_list"] else None)
            mean_mae_vx = (float(np.mean(stats["mae_vx_list"]))
                           if stats["mae_vx_list"] else None)
            mean_mae_vy = (float(np.mean(stats["mae_vy_list"]))
                           if stats["mae_vy_list"] else None)

            # Cumulative MAE
            mean_cum_mae_vxy = (float(np.mean(stats["cum_mae_vxy_list"]))
                                if stats["cum_mae_vxy_list"] else None)
            mean_cum_mae_vx = (float(np.mean(stats["cum_mae_vx_list"]))
                               if stats["cum_mae_vx_list"] else None)
            mean_cum_mae_vy = (float(np.mean(stats["cum_mae_vy_list"]))
                               if stats["cum_mae_vy_list"] else None)

            # z-vel metrics
            mean_z_rmse = (float(np.mean(stats["z_rmse_list"]))
                           if stats["z_rmse_list"] else None)
            mean_z_mae = (float(np.mean(stats["z_mae_list"]))
                          if stats["z_mae_list"] else None)

            # Survivability
            survival_pol = (
                (stats["survived_steps_sum"] / stats["total_steps_sum"]) * 100.0
                if stats["total_steps_sum"] > 0 else 0.0
            )

            pol_data[pol] = {
                # RMSE
                "mean_rmse_total_vxy": mean_rmse_vxy,
                "mean_rmse_vx": mean_rmse_vx,
                "mean_rmse_vy": mean_rmse_vy,
                # MAE
                "mean_mae_total_vxy": mean_mae_vxy,
                "mean_mae_vx": mean_mae_vx,
                "mean_mae_vy": mean_mae_vy,
                # Cumulative MAE
                "mean_cumulative_mae_total_vxy": mean_cum_mae_vxy,
                "mean_cumulative_mae_vx": mean_cum_mae_vx,
                "mean_cumulative_mae_vy": mean_cum_mae_vy,
                # z-vel
                "mean_z_vel_rmse_mpc_vs_policy": mean_z_rmse,
                "mean_z_vel_mae_mpc_vs_policy": mean_z_mae,
                # Survivability
                "survivability_percent": survival_pol
            }

        if pol_data:
            seq_summary["Policies"] = pol_data

        final_save = {"sequence_summary": seq_summary}

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        with open(output_json_path, "w") as f:
            json.dump(final_save, f, indent=4)
        print(f"Summary saved to: {output_json_path}")


# ------------------ Filtering and Printing Functions ------------------

def print_episodes_081059_better_than_nom(summary_json_path):
    """
    Prints the combinations of episodes where the '081059' policy's survivability
    is greater than the nominal MPC ('Nom') survivability.

    :param summary_json_path: Path to the aggregated summary JSON file.
    """
    # Check if the summary JSON exists
    if not os.path.isfile(summary_json_path):
        print(f"Summary JSON not found at: {summary_json_path}")
        return

    # Load the summary JSON
    with open(summary_json_path, 'r') as f:
        summary = json.load(f)

    sequence_summary = summary.get("sequence_summary", {})
    episodes = sequence_summary.get("episodes", [])

    # Iterate through each episode
    for ep in episodes:
        nom_surv = ep.get("Nominal", {}).get("survivability_percent", 0)
        pol_surv = ep.get("Policies", {}).get("081059", {}).get("survivability_percent", 0)

        if pol_surv > nom_surv:
            episode_index = ep.get("episode_index", "Unknown")
            sim_time = ep.get("sim_time", "Unknown")
            comb = ep.get("comb", [])

            print(f"Episode {episode_index} - Sim Time: {sim_time}s")
            print("Combination of Sub-Combinations:")
            for subcombo in comb:
                print(subcombo)
            print(f"'081059' Survivability: {pol_surv:.2f}% > 'Nom' Survivability: {nom_surv:.2f}%\n")


def print_episodes_with_vx0_vy0(summary_json_path):
    """
    Prints the combinations of episodes where any sub-combination has commanded vx=0 and vy=0.

    :param summary_json_path: Path to the aggregated summary JSON file.
    """
    # Check if the summary JSON exists
    if not os.path.isfile(summary_json_path):
        print(f"Summary JSON not found at: {summary_json_path}")
        return

    # Load the summary JSON
    with open(summary_json_path, 'r') as f:
        summary = json.load(f)

    sequence_summary = summary.get("sequence_summary", {})
    episodes = sequence_summary.get("episodes", [])

    # Iterate through each episode
    for ep in episodes:
        comb = ep.get("comb", [])
        has_vx0_vy0 = False

        # Check each sub-combo in the episode
        for subcombo in comb:
            # Assuming subcombo structure is [g, vx, vy, w]
            if len(subcombo) >= 3:
                vx = subcombo[1]
                vy = subcombo[2]
                if vx == 0 and vy == 0:
                    has_vx0_vy0 = True
                    break

        if has_vx0_vy0:
            episode_index = ep.get("episode_index", "Unknown")
            sim_time = ep.get("sim_time", "Unknown")
            print(f"Episode {episode_index} - Sim Time: {sim_time}s")
            print("Combination of Sub-Combinations:")
            for subcombo in comb:
                print(subcombo)
            print("Contains a sub-combination with vx=0 and vy=0.\n")
            
# ------------------ Example Usage ------------------

if __name__ == "__main__":
    # 1) Generate 50 episodes, each with 4 random sub-combos
    random.seed(42)   # Define input ranges
    inside = True   
    vx_in = np.arange(-0.1, 0.6, 0.1)
    vy_in = np.arange(-0.3, 0.4, 0.1)

    # Round the grid values
    vx_in = np.around(vx_in, 1)
    vy_in = np.around(vy_in, 1)

    # Generate combinations for input
    comb_in = list(itertools.product([1, 0], vx_in, vy_in, [0]))
    comb_in = [tuple(c) for c in comb_in]  # Convert to tuples for set operations

    # Define output ranges
    vx_out = np.arange(-0.2, 1.1, 0.1)
    vy_out = np.arange(-0.6, 0.7, 0.1)

    # Round the grid values
    vx_out = np.around(vx_out, 1)
    vy_out = np.around(vy_out, 1)

    # Generate combinations for output
    comb_out = list(itertools.product([1, 0], vx_out, vy_out, [0]))
    comb_out = [tuple(c) for c in comb_out]  # Convert to tuples for set operations

    # Convert lists to sets
    set_in = set(comb_in)
    set_out = set(comb_out)

    # Compute the complementary set
    if inside:
        combinations = set_in
    else:
        combinations = list(set_out - set_in)
    # (Optional) If you need the result as a list of lists
    all_combinations = [list(c) for c in combinations]

    comb = []
    sim_time_list = []
    for _ in range(50):
        available = all_combinations.copy()
        episode_subcombos = []
        for _ in range(4):
            element = random.choice(available)
            episode_subcombos.append(element)
            available.remove(element)

        st = round(random.uniform(1.0, 1.9), 1)
        st+=5
        comb.append(episode_subcombos)
        sim_time_list.append(st)
    
    
    # 2) Define policies
    policies = ["Nom", "081059", "131519", "131330"]

    # 3) Specify the folder where JSON results are located
    results_folder = "/home/federico/biconmp_mujoco/project/results/"

    # 4) Instantiate and run the aggregator
    aggregator = SequenceResultsAggregator(
        comb=comb,
        sim_time_list=sim_time_list,
        policies=policies,
        results_folder=results_folder
    )
    aggregator.aggregate()

    # 5) Save the aggregated summary
    output_json_path = os.path.join(results_folder, "tracking_perf_sequence_summary444IN.json")
    aggregator.save_summary(output_json_path)

    # 6) Print episodes where '081059' survives more than 'Nom'
    print("\n--- Episodes where '081059' Survivability > 'Nom' ---\n")
    print_episodes_081059_better_than_nom(output_json_path)

    # 7) Print episodes where any subcombo has vx=0 and vy=0
    print("\n--- Episodes with Sub-Combination Commanded vx=0 and vy=0 ---\n")
    print_episodes_with_vx0_vy0(output_json_path)
