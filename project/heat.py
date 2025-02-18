def plot_cmd_vel_rmse_heatmap(error_metric="mae", region="in"):
 
    """
    Plots heatmaps of either RMSE or MAE for commanded velocities (vx, vy):
      - XY total error (rmse_total_vxy or mae_total_vxy)
      - VX-only error (rmse_vx or mae_vx)
      - VY-only error (rmse_vy or mae_vy)
      - Z velocity error (for learned policies)
      - Cumulative XY error (MAE only)
      - Survived steps

    When region="in", bounding box is [-0.1..0.5] x [-0.3..0.3].
    When region="out", bounding box is [-0.2..1.0] x [-0.6..0.6],
      and we highlight the smaller box in red on the plots;
      mean/std stats are computed only from points outside that smaller region.
    """
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches

    # -------------------------------------------------------------------------
    # 1) Imports from compareConf / Global setup
    # -------------------------------------------------------------------------
    from compareConf import policy as gpol
    from compareConf import comb as gcomb
    from compareConf import sim_time as gtime
    from compareConf import multi as gmulti

    # Decide bounding box based on region
    if region.lower() == "in":
        x_min, x_max = -0.1, 0.5
        y_min, y_max = -0.3, 0.3
    elif region.lower() == "out":
        x_min, x_max = -0.2, 1.0
        y_min, y_max = -0.6, 0.6
    else:
        raise ValueError("region must be either 'in' or 'out'")

    # Smaller box for excluding stats if region="out"
    SMALL_X_MIN, SMALL_X_MAX = -0.1, 0.5
    SMALL_Y_MIN, SMALL_Y_MAX = -0.3, 0.3

    def is_in_bounding_box(vx, vy):
        return (x_min <= vx <= x_max) and (y_min <= vy <= y_max)

    def is_in_small_box(vx, vy):
        return (SMALL_X_MIN <= vx <= SMALL_X_MAX) and (SMALL_Y_MIN <= vy <= SMALL_Y_MAX)

    # Decide policies
    if gpol == "all":
        policies_to_do = ["Nom", "081059", "131519", "131330", "111351"]
    else:
        policies_to_do = [gpol]

    current_time = str(gtime)
    combos = gcomb
    unique_gaits = sorted(set(c[0] for c in combos))

    results_folder = "/home/federico/biconmp_mujoco/project/results/"

    # -------------------------------------------------------------------------
    # 1.5) Select keys based on error_metric
    # -------------------------------------------------------------------------
    error_metric = error_metric.lower()
    if error_metric == "rmse":
        main_err_key   = "rmse_total_vxy"      # XY total
        main_err_key_vx = "rmse_vx"            # VX-only
        main_err_key_vy = "rmse_vy"            # VY-only
        z_vel_err_key  = "z_vel_rmse_mpc_vs_policy"
        err_label      = "RMSE total v_xy"
        vx_label       = "RMSE vx"
        vy_label       = "RMSE vy"
        out_label      = "rmse"
        # Typically no 'cumulative_rmse_total_vxy' in your JSON => set to None
        cumulative_key = None
    elif error_metric == "mae":
        main_err_key   = "mae_total_vxy"       # XY total
        main_err_key_vx = "mae_vx"             # VX-only
        main_err_key_vy = "mae_vy"             # VY-only
        z_vel_err_key  = "z_vel_mae_mpc_vs_policy"
        err_label      = "MAE total v_xy"
        vx_label       = "MAE vx"
        vy_label       = "MAE vy"
        out_label      = "mae"
        # For MAE, we do have a "cumulative_mae_total_vxy"
        cumulative_key = "cumulative_mae_total_vxy"
    else:
        raise ValueError(f"Unsupported error_metric={error_metric}. Use 'rmse' or 'mae'.")

    # Survived steps key
    survive_key = "survived_timesteps"

    # Gait naming
    gait_name_map = {0: "Trot", 1: "Jump"}

    # Policy naming
    policy_titles = {
        "Nom":    "BiConMP",
        "081059": "Hybrid 081059",
        "131519": "Velocity conditioned 131519",
        "131330": "Contact conditioned 131330"
    }

    # -------------------------------------------------------------------------
    # 2) Loop over each gait
    # -------------------------------------------------------------------------
    for gait_value in unique_gaits:
        combos_for_this_gait = [c for c in combos if c[0] == gait_value]
        if not combos_for_this_gait:
            print(f"No combos found for gait={gait_value}. Skipping.")
            continue

        # Gather all vx, vy in the bounding box
        vx_values_all = sorted(set(c[1] for c in combos_for_this_gait))
        vy_values_all = sorted(set(c[2] for c in combos_for_this_gait))

        vx_values = [v for v in vx_values_all if x_min <= v <= x_max]
        vy_values = [v for v in vy_values_all if y_min <= v <= y_max]

        if not vx_values or not vy_values:
            print(f"No (vx, vy) within bounding box for gait={gait_value}. Skipping.")
            continue

        # Prepare a dictionary for each policy => arrays
        #   (err_arr, surv_arr, zerr_arr, cum_err_arr, vxerr_arr, vyerr_arr)
        # We'll store each as 2D arrays of shape [len(vy_values), len(vx_values)]
        policy_data = {}

        for pol in policies_to_do:
            # Nominal uses "mpc_vs_commanded"; learned uses "policy_vs_commanded"
            if pol == "Nom":
                dict_key = "mpc_vs_commanded"
                suffix   = "_cmd_vel_comparison.json"
                prefix   = "tracking_perf_Nom"
            else:
                dict_key = "policy_vs_commanded"
                suffix   = "_cmd_vel_comparison.json"
                prefix   = f"tracking_perf_{pol}"

            # Initialize arrays
            err_arr    = np.full((len(vy_values), len(vx_values)), np.nan)  # total XY error
            surv_arr   = np.full((len(vy_values), len(vx_values)), np.nan)
            zerr_arr   = np.full((len(vy_values), len(vx_values)), np.nan)  # Z error
            cum_arr    = np.full((len(vy_values), len(vx_values)), np.nan)  # cumulative XY
            vxerr_arr  = np.full((len(vy_values), len(vx_values)), np.nan)  # vx-only error
            vyerr_arr  = np.full((len(vy_values), len(vx_values)), np.nan)  # vy-only error

            for combo_item in combos_for_this_gait:
                gait, vx, vy, wz = combo_item
                if not is_in_bounding_box(vx, vy):
                    continue

                combo_str = str([combo_item])
                json_filename = f"{prefix}{combo_str}{current_time}{suffix}"
                json_path = os.path.join(results_folder, json_filename)

                if not os.path.isfile(json_path):
                    continue

                with open(json_path, "r") as jf:
                    data = json.load(jf)

                if dict_key not in data:
                    continue

                sub_dict = data[dict_key]

                main_err_val_xy = sub_dict.get(main_err_key, np.nan)
                surv_val        = sub_dict.get(survive_key, np.nan)

                # Separate vx/ vy errors
                vx_err_val = sub_dict.get(main_err_key_vx, np.nan)
                vy_err_val = sub_dict.get(main_err_key_vy, np.nan)

                # Z-axis error for learned policies
                z_err_val = np.nan
                if pol != "Nom" and z_vel_err_key in data:
                    z_err_val = data[z_vel_err_key]

                # Cumulative error (only if MAE & found in sub_dict)
                if cumulative_key and cumulative_key in sub_dict and surv_val > 0:
                    c_val = sub_dict[cumulative_key]
                else:
                    c_val = np.nan

                # Indices
                try:
                    i_vy = vy_values.index(vy)
                    j_vx = vx_values.index(vx)
                except ValueError:
                    continue

                # Fill arrays
                err_arr[i_vy, j_vx]  = main_err_val_xy
                surv_arr[i_vy, j_vx] = surv_val
                zerr_arr[i_vy, j_vx] = z_err_val
                cum_arr[i_vy, j_vx]  = c_val
                vxerr_arr[i_vy, j_vx] = vx_err_val
                vyerr_arr[i_vy, j_vx] = vy_err_val

            policy_data[pol] = (err_arr, surv_arr, zerr_arr, cum_arr, vxerr_arr, vyerr_arr)

        # ---------------------------------------------------------------------
        # 3) Determine color scales
        # ---------------------------------------------------------------------
        def valid_vals(arr): return arr[~np.isnan(arr)]
        # We'll gather from all policies to get a global min-max for color normalization

        all_err_values_xy = []
        all_surv_values   = []
        all_zerr_values   = []
        all_cum_values    = []
        all_vxerr_values  = []
        all_vyerr_values  = []

        for pol in policies_to_do:
            earr, sarr, zarr, carr, vxarr, vyarr = policy_data[pol]
            all_err_values_xy.append(valid_vals(earr))
            all_surv_values.append(valid_vals(sarr))
            if pol != "Nom":
                all_zerr_values.append(valid_vals(zarr))
            if cumulative_key:
                all_cum_values.append(valid_vals(carr))
            all_vxerr_values.append(valid_vals(vxarr))
            all_vyerr_values.append(valid_vals(vyarr))

        def flatten(arr_list):
            arr_list_nonempty = [a for a in arr_list if a.size > 0]
            return np.concatenate(arr_list_nonempty) if arr_list_nonempty else np.array([])

        err_all_xy = flatten(all_err_values_xy)
        surv_all   = flatten(all_surv_values)
        zerr_all   = flatten(all_zerr_values)
        cum_all    = flatten(all_cum_values)
        vxerr_all  = flatten(all_vxerr_values)
        vyerr_all  = flatten(all_vyerr_values)

        # If no XY error data, skip
        if err_all_xy.size == 0:
            print(f"No valid XY {error_metric.upper()} data for gait={gait_value}. Skipping plots.")
            continue

        # XY error norm
        err_min_xy, err_max_xy = np.min(err_all_xy), np.max(err_all_xy)
        if np.isclose(err_min_xy, err_max_xy):
            eps = 0.01 if err_min_xy == 0 else abs(0.05 * err_min_xy)
            err_min_xy -= eps
            err_max_xy += eps
        err_norm_xy = mcolors.Normalize(vmin=err_min_xy, vmax=err_max_xy)

        # Surv norm
        if surv_all.size > 0:
            s_min, s_max = np.min(surv_all), np.max(surv_all)
            if np.isclose(s_min, s_max):
                s_min -= 0.5
                s_max += 0.5
        else:
            s_min, s_max = (0, 1)
        surv_norm = mcolors.Normalize(vmin=s_min, vmax=s_max)

        # Z error norm
        have_zerr = (zerr_all.size > 0)
        if have_zerr:
            z_min, z_max = np.min(zerr_all), np.max(zerr_all)
            if np.isclose(z_min, z_max):
                eps_z = 0.01 if z_min == 0 else abs(0.05 * z_min)
                z_min -= eps_z
                z_max += eps_z
            zerr_norm = mcolors.Normalize(vmin=z_min, vmax=z_max)
        else:
            zerr_norm = None

        # Cumulative norm
        have_cum = (cum_all.size > 0)
        if have_cum:
            c_min, c_max = np.min(cum_all), np.max(cum_all)
            if np.isclose(c_min, c_max):
                eps_c = 0.01 if c_min == 0 else abs(0.05 * c_min)
                c_min -= eps_c
                c_max += eps_c
            cum_scale = mcolors.Normalize(vmin=c_min, vmax=c_max)
        else:
            cum_scale = None

        # VX error norm
        if vxerr_all.size > 0:
            vx_min, vx_max = np.min(vxerr_all), np.max(vxerr_all)
            if np.isclose(vx_min, vx_max):
                eps_vx = 0.01 if vx_min == 0 else abs(0.05 * vx_min)
                vx_min -= eps_vx
                vx_max += eps_vx
            vxerr_norm = mcolors.Normalize(vmin=vx_min, vmax=vx_max)
            have_vxerr = True
        else:
            have_vxerr = False
            vxerr_norm = None

        # VY error norm
        if vyerr_all.size > 0:
            vy_min, vy_max = np.min(vyerr_all), np.max(vyerr_all)
            if np.isclose(vy_min, vy_max):
                eps_vy = 0.01 if vy_min == 0 else abs(0.05 * vy_min)
                vy_min -= eps_vy
                vy_max += eps_vy
            vyerr_norm = mcolors.Normalize(vmin=vy_min, vmax=vy_max)
            have_vyerr = True
        else:
            have_vyerr = False
            vyerr_norm = None

        # ---------------------------------------------------------------------
        # 4) Plot XY Error Heatmap
        # ---------------------------------------------------------------------
        gait_title = gait_name_map.get(gait_value, f"Gait {gait_value}")
        n_policies = len(policies_to_do)

        fig_err, axes_err = plt.subplots(1, n_policies, figsize=(4*n_policies+2, 4), squeeze=False)
        for i_pol, pol in enumerate(policies_to_do):
            earr, _, _, _, _, _ = policy_data[pol]
            ax_err = axes_err[0, i_pol]
            im = ax_err.imshow(
                earr,
                origin='lower',
                cmap='viridis',
                norm=err_norm_xy,
                extent=[x_min, x_max, y_min, y_max],
                aspect='auto'
            )
            ax_err.set_title(policy_titles.get(pol, pol), fontsize=10)
            ax_err.set_xlabel("vx")
            ax_err.set_ylabel("vy")

            # If region="out", highlight the smaller box in red
            if region.lower() == "out":
                rect = patches.Rectangle(
                    (SMALL_X_MIN, SMALL_Y_MIN),
                    (SMALL_X_MAX - SMALL_X_MIN),
                    (SMALL_Y_MAX - SMALL_Y_MIN),
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax_err.add_patch(rect)

        cbar_ax_err = fig_err.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_obj_err = plt.cm.ScalarMappable(norm=err_norm_xy, cmap='viridis')
        fig_err.colorbar(cbar_obj_err, cax=cbar_ax_err, label=err_label)

        fig_err.suptitle(
            f"{gait_title} - XY {error_metric.upper()} (region={region})\n"
            f"Simulation time {current_time} s",
            fontsize=12
        )
        fig_err.tight_layout(rect=[0, 0, 0.9, 1])

        out_err_png = os.path.join(
            results_folder,
            f"heatmap_cmd_vel_{out_label}_{region}_gait{gait_value}_{current_time}.png"
        )
        plt.savefig(out_err_png, dpi=150)
        plt.show()
        print(f"Saved XY {error_metric.upper()} heatmap => {out_err_png}")

        # ---------------------------------------------------------------------
        # 4b) Plot VX Error Heatmap (if data exist)
        # ---------------------------------------------------------------------
        if have_vxerr:
            fig_vx, axes_vx = plt.subplots(1, n_policies, figsize=(4*n_policies+2, 4), squeeze=False)
            for i_pol, pol in enumerate(policies_to_do):
                _, _, _, _, vxarr, _ = policy_data[pol]
                ax_vx = axes_vx[0, i_pol]
                ax_vx.imshow(
                    vxarr,
                    origin='lower',
                    cmap='plasma',
                    norm=vxerr_norm,
                    extent=[x_min, x_max, y_min, y_max],
                    aspect='auto'
                )
                ax_vx.set_title(policy_titles.get(pol, pol), fontsize=10)
                ax_vx.set_xlabel("vx")
                ax_vx.set_ylabel("vy")
                if region.lower() == "out":
                    rect = patches.Rectangle(
                        (SMALL_X_MIN, SMALL_Y_MIN),
                        (SMALL_X_MAX - SMALL_X_MIN),
                        (SMALL_Y_MAX - SMALL_Y_MIN),
                        fill=False,
                        edgecolor='red',
                        linewidth=2
                    )
                    ax_vx.add_patch(rect)

            cbar_ax_vx = fig_vx.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar_obj_vx = plt.cm.ScalarMappable(norm=vxerr_norm, cmap='plasma')
            fig_vx.colorbar(cbar_obj_vx, cax=cbar_ax_vx, label=vx_label)

            fig_vx.suptitle(
                f"{gait_title} - VX {error_metric.upper()} (region={region})\n"
                f"Simulation time {current_time} s",
                fontsize=12
            )
            fig_vx.tight_layout(rect=[0, 0, 0.9, 1])

            out_vx_png = os.path.join(
                results_folder,
                f"heatmap_cmd_vel_vx_{out_label}_{region}_gait{gait_value}_{current_time}.png"
            )
            plt.savefig(out_vx_png, dpi=150)
            plt.show()
            print(f"Saved VX {error_metric.upper()} heatmap => {out_vx_png}")

        # ---------------------------------------------------------------------
        # 4c) Plot VY Error Heatmap (if data exist)
        # ---------------------------------------------------------------------
        if have_vyerr:
            fig_vy, axes_vy = plt.subplots(1, n_policies, figsize=(4*n_policies+2, 4), squeeze=False)
            for i_pol, pol in enumerate(policies_to_do):
                _, _, _, _, _, vyarr = policy_data[pol]
                ax_vy = axes_vy[0, i_pol]
                ax_vy.imshow(
                    vyarr,
                    origin='lower',
                    cmap='plasma',
                    norm=vyerr_norm,
                    extent=[x_min, x_max, y_min, y_max],
                    aspect='auto'
                )
                ax_vy.set_title(policy_titles.get(pol, pol), fontsize=10)
                ax_vy.set_xlabel("vx")
                ax_vy.set_ylabel("vy")

                if region.lower() == "out":
                    rect = patches.Rectangle(
                        (SMALL_X_MIN, SMALL_Y_MIN),
                        (SMALL_X_MAX - SMALL_X_MIN),
                        (SMALL_Y_MAX - SMALL_Y_MIN),
                        fill=False,
                        edgecolor='red',
                        linewidth=2
                    )
                    ax_vy.add_patch(rect)

            cbar_ax_vy = fig_vy.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar_obj_vy = plt.cm.ScalarMappable(norm=vyerr_norm, cmap='plasma')
            fig_vy.colorbar(cbar_obj_vy, cax=cbar_ax_vy, label=vy_label)

            fig_vy.suptitle(
                f"{gait_title} - VY {error_metric.upper()} (region={region})\n"
                f"Simulation time {current_time} s",
                fontsize=12
            )
            fig_vy.tight_layout(rect=[0, 0, 0.9, 1])

            out_vy_png = os.path.join(
                results_folder,
                f"heatmap_cmd_vel_vy_{out_label}_{region}_gait{gait_value}_{current_time}.png"
            )
            plt.savefig(out_vy_png, dpi=150)
            plt.show()
            print(f"Saved VY {error_metric.upper()} heatmap => {out_vy_png}")

        # ---------------------------------------------------------------------
        # 5) Plot Cumulative XY Error Heatmap (if MAE + data present)
        # ---------------------------------------------------------------------
        if have_cum and (error_metric == "mae"):
            fig_cum, axes_cum = plt.subplots(1, n_policies, figsize=(4*n_policies+2, 4), squeeze=False)
            for i_pol, pol in enumerate(policies_to_do):
                _, _, _, carr, _, _ = policy_data[pol]
                ax_cum = axes_cum[0, i_pol]
                ax_cum.imshow(
                    carr,
                    origin='lower',
                    cmap='magma',
                    norm=cum_scale,
                    extent=[x_min, x_max, y_min, y_max],
                    aspect='auto'
                )
                ax_cum.set_title(policy_titles.get(pol, pol), fontsize=10)
                ax_cum.set_xlabel("vx")
                ax_cum.set_ylabel("vy")

                if region.lower() == "out":
                    rect = patches.Rectangle(
                        (SMALL_X_MIN, SMALL_Y_MIN),
                        (SMALL_X_MAX - SMALL_X_MIN),
                        (SMALL_Y_MAX - SMALL_Y_MIN),
                        fill=False,
                        edgecolor='red',
                        linewidth=2
                    )
                    ax_cum.add_patch(rect)

            cbar_ax_cum = fig_cum.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar_obj_cum = plt.cm.ScalarMappable(norm=cum_scale, cmap='magma')
            fig_cum.colorbar(cbar_obj_cum, cax=cbar_ax_cum, label="Cumulative MAE (raw)")

            fig_cum.suptitle(
                f"{gait_title} - Cumulative MAE (region={region})\n"
                f"Simulation time {current_time} s",
                fontsize=12
            )
            fig_cum.tight_layout(rect=[0, 0, 0.9, 1])

            out_cum_png = os.path.join(
                results_folder,
                f"heatmap_cmd_vel_cum_mae_{region}_gait{gait_value}_{current_time}.png"
            )
            plt.savefig(out_cum_png, dpi=150)
            plt.show()
            print(f"Saved Cumulative MAE heatmap => {out_cum_png}")

        # ---------------------------------------------------------------------
        # 6) Plot Survived Timesteps Heatmap
        # ---------------------------------------------------------------------
        fig_surv, axes_surv = plt.subplots(1, n_policies, figsize=(4*n_policies+2, 4), squeeze=False)
        for i_pol, pol in enumerate(policies_to_do):
            _, sarr, _, _, _, _ = policy_data[pol]
            ax_s = axes_surv[0, i_pol]
            ax_s.imshow(
                sarr,
                origin='lower',
                cmap='plasma',
                norm=surv_norm,
                extent=[x_min, x_max, y_min, y_max],
                aspect='auto'
            )
            ax_s.set_title(policy_titles.get(pol, pol), fontsize=10)
            ax_s.set_xlabel("vx")
            ax_s.set_ylabel("vy")

            if region.lower() == "out":
                rect = patches.Rectangle(
                    (SMALL_X_MIN, SMALL_Y_MIN),
                    (SMALL_X_MAX - SMALL_X_MIN),
                    (SMALL_Y_MAX - SMALL_Y_MIN),
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax_s.add_patch(rect)

        cbar_ax_surv = fig_surv.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_obj_surv = plt.cm.ScalarMappable(norm=surv_norm, cmap='plasma')
        fig_surv.colorbar(cbar_obj_surv, cax=cbar_ax_surv, label="Survived Timesteps")

        fig_surv.suptitle(
            f"{gait_title} - Survived Steps (region={region})\n"
            f"Simulation time {current_time} s",
            fontsize=12
        )
        fig_surv.tight_layout(rect=[0, 0, 0.9, 1])

        out_surv_png = os.path.join(
            results_folder,
            f"heatmap_cmd_vel_surv_{region}_gait{gait_value}_{current_time}.png"
        )
        plt.savefig(out_surv_png, dpi=150)
        plt.show()
        print(f"Saved Survived Steps heatmap => {out_surv_png}")

        # ---------------------------------------------------------------------
        # 7) Plot Z Error Heatmap (if we have any valid data)
        # ---------------------------------------------------------------------
        if have_zerr:
            # Only learned policies have Z error
            pols_with_z = [p for p in policies_to_do if not np.isnan(policy_data[p][2]).all()]
            if pols_with_z:  # at least one policy has valid Z
                fig_z, axes_z = plt.subplots(1, len(pols_with_z),
                                             figsize=(4*len(pols_with_z)+2, 4), squeeze=False)
                for idx, pol in enumerate(pols_with_z):
                    _, _, zarr, _, _, _ = policy_data[pol]
                    ax_z = axes_z[0, idx]
                    ax_z.imshow(
                        zarr,
                        origin='lower',
                        cmap='cividis',
                        norm=zerr_norm,
                        extent=[x_min, x_max, y_min, y_max],
                        aspect='auto'
                    )
                    ax_z.set_title(policy_titles.get(pol, pol) + f" - Z {error_metric.upper()}", fontsize=10)
                    ax_z.set_xlabel("vx")
                    ax_z.set_ylabel("vy")

                    if region.lower() == "out":
                        rect = patches.Rectangle(
                            (SMALL_X_MIN, SMALL_Y_MIN),
                            (SMALL_X_MAX - SMALL_X_MIN),
                            (SMALL_Y_MAX - SMALL_Y_MIN),
                            fill=False,
                            edgecolor='red',
                            linewidth=2
                        )
                        ax_z.add_patch(rect)

                cbar_ax_z = fig_z.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar_obj_z = plt.cm.ScalarMappable(norm=zerr_norm, cmap='cividis')
                fig_z.colorbar(cbar_obj_z, cax=cbar_ax_z, label=f"Z {error_metric.upper()}")

                fig_z.suptitle(
                    f"{gait_title} - Z {error_metric.upper()} (region={region})\n"
                    f"Simulation time {current_time} s",
                    fontsize=12
                )
                fig_z.tight_layout(rect=[0, 0, 0.9, 1])

                out_z_png = os.path.join(
                    results_folder,
                    f"heatmap_cmd_vel_z_{out_label}_{region}_gait{gait_value}_{current_time}.png"
                )
                plt.savefig(out_z_png, dpi=150)
                plt.show()
                print(f"Saved Z {error_metric.upper()} heatmap => {out_z_png}")

        # ---------------------------------------------------------------------
        # 8) Print Stats
        #
        #    If region="out", we exclude the smaller box from stats.
        #    If region="in", we include everything in the bounding box.
        # ---------------------------------------------------------------------
        print(f"\n=== STATISTICS (region={region}), gait={gait_value} ===")
        for pol in policies_to_do:
            (earr, sarr, zarr, _, vxarr, vyarr) = policy_data[pol]

            # Valid data masks
            valid_xy_mask  = ~np.isnan(earr)
            valid_surv_mask= ~np.isnan(sarr)
            valid_z_mask   = ~np.isnan(zarr)
            valid_vx_mask  = ~np.isnan(vxarr)
            valid_vy_mask  = ~np.isnan(vyarr)

            if region.lower() == "out":
                # Exclude points inside the small region
                inside_small_box = np.zeros_like(earr, dtype=bool)
                for i, vy_ in enumerate(vy_values):
                    for j, vx_ in enumerate(vx_values):
                        if is_in_small_box(vx_, vy_):
                            inside_small_box[i, j] = True
                outside_small_mask = ~inside_small_box

                final_xy_mask  = valid_xy_mask  & outside_small_mask
                final_surv_mask= valid_surv_mask& outside_small_mask
                final_z_mask   = valid_z_mask   & outside_small_mask
                final_vx_mask  = valid_vx_mask  & outside_small_mask
                final_vy_mask  = valid_vy_mask  & outside_small_mask
            else:
                # region="in"
                final_xy_mask  = valid_xy_mask
                final_surv_mask= valid_surv_mask
                final_z_mask   = valid_z_mask
                final_vx_mask  = valid_vx_mask
                final_vy_mask  = valid_vy_mask

            xy_vals  = earr[final_xy_mask]
            surv_vals= sarr[final_surv_mask]
            z_vals   = zarr[final_z_mask] if pol != "Nom" else np.array([])
            vx_vals  = vxarr[final_vx_mask]
            vy_vals  = vyarr[final_vy_mask]

            print(f"\nPolicy '{pol}':")
            # XY
            if xy_vals.size > 0:
                mean_xy = np.mean(xy_vals)
                std_xy  = np.std(xy_vals)
                print(f"   XY {error_metric.upper()}: mean={mean_xy:.4f}, std={std_xy:.4f}")
            else:
                print("   XY Error: no valid data in final region mask.")

            # VX
            if vx_vals.size > 0:
                mean_vx = np.mean(vx_vals)
                std_vx  = np.std(vx_vals)
                print(f"   VX {error_metric.upper()}: mean={mean_vx:.4f}, std={std_vx:.4f}")
            else:
                print("   VX Error: no valid data in final region mask.")

            # VY
            if vy_vals.size > 0:
                mean_vy = np.mean(vy_vals)
                std_vy  = np.std(vy_vals)
                print(f"   VY {error_metric.upper()}: mean={mean_vy:.4f}, std={std_vy:.4f}")
            else:
                print("   VY Error: no valid data in final region mask.")

            # Z
            if pol != "Nom":
                if z_vals.size > 0:
                    mean_z = np.mean(z_vals)
                    std_z  = np.std(z_vals)
                    print(f"   Z {error_metric.upper()}: mean={mean_z:.4f}, std={std_z:.4f}")
                else:
                    print("   Z Error: no valid data (or missing) in final region mask.")
            else:
                print("   Z Error: Not applicable for 'Nom' policy.")

            # Survival
            if surv_vals.size > 0:
                mean_surv = np.mean(surv_vals)
                print(f"   Survivability: mean={mean_surv:.1f} steps")
            else:
                print("   Survivability: no valid data in final region mask.")
        print("----------------------------------------------------\n")

plot_cmd_vel_rmse_heatmap(error_metric="mae", region="out")      