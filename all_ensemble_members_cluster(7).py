"""
lagranto_ensemble_clustering.py

Combines lagranto forward-trajectory files from multiple ensemble members into
a single dataset, verifies they are structurally consistent first, then runs
the same K-means clustering (by location + change in height) used in the
single-member `lagranto_analysis` script and produces the same cluster plots
-- but for the whole ensemble at once as well as two subsets of ensemble members.
All plots are written to `all_trajectories_folder`.

Usage: python all_ensemble_members_cluster.py --plot-all --output-dir 2023010400_all_trajectories

"""

import os
import re
import glob
import warnings
import datetime as dt
from datetime import timedelta
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION -- adjust these as needed
# ============================================================================

# Base directory containing all member_lagranto folders
BASE_DIR = "/nfs/tornlab_rit/hs559346"

# Init time string as it appears in the trajectory filenames
INIT_TIME = "2023010300"

# Find trajectory files
FILE_PATTERN = os.path.join(BASE_DIR, "**", f"forward_member*_ivt_start_{INIT_TIME}.nc")

# Default output directory if none specified in command line
OUTPUT_DIR = f"{INIT_TIME}_case_combined_trajectory_plots"

# MEMBER GROUPS
##### -------------------- #####
MEMBER_GROUPS = {
    "1": [4,12,36,37,48,66,72,87,90],   # <-- member ids for group 1
    "2": [2,23,26,33,42,47,50,61,83],   # <-- member ids for group 2
}
##### -------------------- #####

# Domain used to discard trajectories that leave the region of interest
# (same as the single-member script).
STARTLAT, ENDLAT = 20, 70
STARTLON, ENDLON = 150, 330

# Clustering parameters (same as the single-member script).
Z_WEIGHT = 0.8
K_VALUES = [3, 4, 5, 6]     # cluster counts to compute stable/aligned labels for
K_TO_PLOT = [5]             # which of the above to actually generate plots for

# Map extent used for the cluster trajectory map plot.
MAP_EXTENT = [-85, -145, 25, 60]

# Variables that are allowed to be missing from some member files without
# failing the consistency check (e.g. some runs don't have IVT computed).
# Members missing an optional variable simply get NaN for it after combining.
OPTIONAL_VARS = {"ivt"}

# Internal key used for the optional "all members together" set of plots.
ALL_KEY = "all_members"

def parse_group_specs(specs, parser):
    """Turn ['1=1,3,5', '2=2,4,6'] into {'1': [1, 3, 5], '2': [2, 4, 6]}."""
    groups = {}
    for spec in specs:
        if "=" not in spec:
            parser.error(f"--group expects NAME=ID,ID,... but got '{spec}'")
        name, ids = spec.split("=", 1)
        try:
            groups[name.strip()] = [int(x) for x in ids.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--group member ids must be integers, got '{spec}'")
    return groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine and cluster lagranto trajectories for a set of ensemble members."
    )
    parser.add_argument(
        "--plot-all", action="store_true",
        help="Also make one set of plots containing every member (folder 'all_members').",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Folder to write cluster plots to (default: '{OUTPUT_DIR}').",
        )
    args = parser.parse_args()
    groups = (
        {str(name): list(ids) for name, ids in MEMBER_GROUPS.items()}
    )
    return args, groups
 
 
args, groups = parse_args()
 
if args.output_dir:
    OUTPUT_DIR = args.output_dir


# ============================================================================
# 1. Discover ensemble member trajectory files
# ============================================================================

def discover_member_files(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No trajectory files found matching pattern:\n  {pattern}\n"
            "Check BASE_DIR / INIT_TIME, or that the files are reachable."
        )
    return files


def member_id_from_path(path):
    match = re.search(r"forward_member(\d+)_ivt_start", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse a member id out of filename: {path}")
    return int(match.group(1))


member_files = discover_member_files(FILE_PATTERN)

found_ids = {member_id_from_path(f) for f in member_files}
 
print(f"Found {len(member_files)} ensemble member trajectory file(s):")
for f in member_files:
    print(f"  {f}")
 
 
# ============================================================================
# 1b. Keep only ensemble members found in the groups
# ============================================================================
 
def validate_groups(groups, found_ids):
    if not groups or any(len(ids) == 0 for ids in groups.values()):
        raise ValueError(
            "Every member group needs at least one member id. Fill in MEMBER_GROUPS "
            "in the config block, or pass e.g. --group 1=1,3,5 --group 2=2,4,6.\n"
            f"Discovered member ids: {sorted(found_ids)}"
        )
 
    for name, ids in groups.items():
        missing = set(ids) - found_ids
        if missing:
            raise FileNotFoundError(
                f"Group '{name}' lists member id(s) that were not discovered: "
                f"{sorted(missing)}\nDiscovered member ids: {sorted(found_ids)}"
            )
 
    owner = {}
    for name, ids in groups.items():
        for m in ids:
            if m in owner:
                print(f"WARNING: member {m} is in both group '{owner[m]}' and group "
                      f"'{name}' -- it will be plotted in both.")
            owner.setdefault(m, name)
 
    unassigned = sorted(found_ids - set(owner))
    if unassigned:
        print(f"NOTE: member(s) {unassigned} are in no group. They are not used for clustering.")
 
 
validate_groups(groups, found_ids)

# Keep ONLY members that belong to at least one group. Everything downstream
# (consistency check, combining, domain filter, clustering) sees just these.
grouped_ids = {m for ids in groups.values() for m in ids}
member_files = [f for f in member_files if member_id_from_path(f) in grouped_ids]
 
print(f"\nUsing {len(member_files)} grouped member file(s) for clustering:")
for f in member_files:
    print(f"  {f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 2. Verify all members share the same dimensions and variables
# ============================================================================

def check_consistency(files, optional_vars=OPTIONAL_VARS):
    """Check that every trajectory file has the same non-ntra dimensions and
    the same set of REQUIRED data variables as the first file. `ntra` (number
    of trajectories) is allowed to differ between members, and any variable
    listed in `optional_vars` is allowed to be present in some files and
    absent in others without being treated as an inconsistency."""
    ref_ds = xr.open_dataset(files[0])
    ref_dims = {d: n for d, n in ref_ds.sizes.items() if d != "ntra"}
    ref_required_vars = set(ref_ds.data_vars) - optional_vars
    ref_ds.close()
 
    problems = []
    for f in files[1:]:
        ds_check = xr.open_dataset(f)
        dims = {d: n for d, n in ds_check.sizes.items() if d != "ntra"}
        required_vars = set(ds_check.data_vars) - optional_vars
 
        if dims != ref_dims:
            problems.append(f"{f}\n    dims {dims} != reference {ref_dims}")
        if required_vars != ref_required_vars:
            missing = ref_required_vars - required_vars
            extra = required_vars - ref_required_vars
            problems.append(
                f"{f}\n    missing required vars: {sorted(missing)}\n    "
                f"extra required vars: {sorted(extra)}"
            )
        ds_check.close()
 
    if problems:
        raise ValueError(
            "Trajectory files are NOT consistent across ensemble members:\n\n"
            + "\n\n".join(problems)
        )
 
    print("All member files share the same dimensions (aside from 'ntra') and required variables.")
    return ref_dims, ref_required_vars
 
 
check_consistency(member_files)


# ============================================================================
# 3. Combine all members into a single dataset along the 'ntra' dimension
# ============================================================================

member_datasets = []
member_id_chunks = []
for f in member_files:
    mid = member_id_from_path(f)
    ds_m = xr.open_dataset(f)
    # Tag each trajectory with which ensemble member it came from. This is what
    # lets us split the jointly-clustered trajectories back into groups later.
    ids_this_file = np.full(ds_m.sizes["ntra"], mid)
    ds_m = ds_m.assign_coords(member_id=("ntra", ids_this_file))
    member_datasets.append(ds_m)
    member_id_chunks.append(ids_this_file)

ds = xr.concat(member_datasets, dim="ntra", data_vars="minimal")
for ds_m in member_datasets:
    ds_m.close()

# One member id per trajectory in the combined dataset (same order as ds).
traj_member_ids = np.concatenate(member_id_chunks)
assert len(traj_member_ids) == ds.sizes["ntra"], "member id bookkeeping does not match combined ntra"

n_members = len(member_files)
print(f"Combined dataset: {ds.sizes['ntra']} total trajectories across {n_members} members.")


# ============================================================================
# 4. Domain filtering (same logic as the single-member script)
# ============================================================================

all_indices = np.arange(ds.sizes["ntra"])


def within_domain(ds, indices, startlat, endlat, startlon, endlon):
    lat = ds.lat[:, indices].values
    lon = ds.lon[:, indices].values
    outside = (
        (lat < startlat) | (lat > endlat) |
        (lon < startlon) | (lon > endlon)
    )
    return ~outside.any(axis=0)


inside_mask = within_domain(ds, all_indices, STARTLAT, ENDLAT, STARTLON, ENDLON)
print(f"Trajectories removed (left domain): {(~inside_mask).sum()}")
print(f"Trajectories remaining: {inside_mask.sum()}")

all_indices = all_indices[inside_mask]

# Member id of each KEPT trajectory. Position i here lines up with labels[i]
# and with all_indices[i] -- this is the array used to split into groups.
kept_member_ids = traj_member_ids[all_indices]


# ============================================================================
# 5. Derived PV tendency variables (used by several cluster time series plots)
# ============================================================================

ds["total_diabatic_tendency"] = (
    ds.acc_depv_dt_lw_isobaric + ds.acc_depv_dt_sw_isobaric + ds.acc_depv_dt_bl_isobaric
    + ds.acc_depv_dt_cu_isobaric + ds.acc_depv_dt_mp_isobaric
)
ds["latent_heat_release"] = ds.acc_depv_dt_cu_isobaric + ds.acc_depv_dt_mp_isobaric
ds["total_diabatic_tendency_with_friction"] = (
    ds.acc_depv_dt_lw_isobaric + ds.acc_depv_dt_sw_isobaric + ds.acc_depv_dt_bl_isobaric
    + ds.acc_depv_dt_cu_isobaric + ds.acc_depv_dt_mp_isobaric + ds.acc_depv_dt_fric_bl_isobaric
)


# ============================================================================
# 6. Reference time for plot titles
# ============================================================================

starting_time = dt.datetime(
    ds.attrs["ref_year"], ds.attrs["ref_month"], ds.attrs["ref_day"],
    ds.attrs["ref_hour"], ds.attrs["ref_min"],
)
traj_init_time = starting_time + timedelta(hours=24)
formatted_time = traj_init_time.strftime("%Y-%m-%d %H:%M UTC")
time_vals = ds.time.values - 24


# ============================================================================
# 7. Feature extraction for clustering: location + change in height
# ============================================================================

def flatten(ds, indices, z_weight):
    """Cluster on location (first 36 hrs, skipping the first 2 timesteps)
    plus each trajectory's height change relative to its starting height."""
    lat = ds.lat[2:36, indices].values
    lon = ds.lon[2:36, indices].values
    z = ds.z[2:36, indices].values

    z_start = ds.z[0, indices].values
    z_rel = z - z_start[np.newaxis, :]

    lat_ref = np.mean(lat)
    lon_ref = np.mean(lon)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat_ref))

    x = (lon - lon_ref) * km_per_deg_lon
    y = (lat - lat_ref) * km_per_deg_lat

    xy = np.concatenate([x.T, y.T], axis=1)
    z_flat = z_rel.T

    xy_scaled = StandardScaler().fit_transform(xy)
    z_scaled = StandardScaler().fit_transform(z_flat) * z_weight

    return np.concatenate([xy_scaled, z_scaled], axis=1)


# ============================================================================
# 8. K-means clustering with cluster-color stability across different k
#    (runs ONCE, on every trajectory from every member)
# ============================================================================

def get_stable_labels(features, k, prev_raw_labels=None, prev_color_map=None):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    raw_labels = km.fit_predict(features)

    if prev_raw_labels is None:
        mean_lats = np.array([
            ds.lat[:, all_indices[raw_labels == c]].values.mean()
            for c in range(k)
        ])
        order = np.argsort(mean_lats)
        remap = np.empty_like(order)
        remap[order] = np.arange(k)
        raw_labels_remapped = remap[raw_labels]
        color_map = {c: c for c in range(k)}
        return raw_labels_remapped, raw_labels_remapped.copy(), color_map

    prev_k = len(np.unique(prev_raw_labels))
    new_k = k

    overlap = np.zeros((new_k, prev_k), dtype=int)
    for i in range(len(raw_labels)):
        overlap[raw_labels[i], prev_raw_labels[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)

    new_color_map = {}
    used_colors = set(prev_color_map.values())
    for new_c, prev_c in zip(row_ind, col_ind):
        new_color_map[new_c] = prev_color_map[prev_c]
    for new_c in range(new_k):
        if new_c not in new_color_map:
            next_color = max(used_colors) + 1
            new_color_map[new_c] = next_color
            used_colors.add(next_color)

    remapped = np.array([new_color_map[label] for label in raw_labels])
    return remapped, raw_labels.copy(), new_color_map


features = flatten(ds, all_indices, Z_WEIGHT)

prev_raw_labels, prev_color_map = None, None
results = {}
for k in K_VALUES:
    labels, raw_labels, color_map = get_stable_labels(
        features, k, prev_raw_labels, prev_color_map
    )
    results[k] = labels
    prev_raw_labels = raw_labels
    prev_color_map = color_map

cluster_cmap = plt.cm.tab10
cluster_colors = [cluster_cmap(i) for i in range(10)]

# ============================================================================
# 9. Split the (already clustered) trajectories into member groups
# ============================================================================
# Each entry in group_positions is a trajectory from that group's members that survived the domain filter.
# No clustering here -- only choose which trajectories to draw for each group.
 
group_positions = {
    name: np.where(np.isin(kept_member_ids, ids))[0]
    for name, ids in groups.items()
}

if args.plot_all:
    group_positions[ALL_KEY] = np.arange(len(all_indices))
 
for name, pos in group_positions.items():
    print(f"  {name}: {len(pos)} trajectories will be plotted")

# ============================================================================
# 10a. Cluster plots (all trajectories, all members combined)
# ============================================================================

kept_cache = {}
 
 
def kept(var):
    """(ntim, n_kept) array of variables for the domain-filtered trajectories.
    Column i corresponds to results[k][i] and all_indices[i]."""
    if var not in kept_cache:
        kept_cache[var] = ds[var].values[:, all_indices]
    return kept_cache[var]

# Display names used on plots and in the cluster counts table in place of the
# raw group keys "1" and "2". The MEMBER_GROUPS keys, group_positions dict,
# and output folder names (group_1/, group_2/) are untouched -- only text
# that is actually drawn or printed uses these.
GROUP_DISPLAY_NAMES = {"1": "weak ridge", "2": "strong ridge"}
 
 
def group_display_name(group):
    """Lowercase display name for a group key, e.g. for use in running text
    ('weak ridge members'). Falls back to 'group {key}' for any key other
    than "1"/"2", so this never silently breaks if a group is renamed/added."""
    return GROUP_DISPLAY_NAMES.get(group, f"group {group}")
 
 
# Display names used on plots and in the cluster counts table in place of the
# raw group keys "1" and "2". The MEMBER_GROUPS keys, group_positions dict,
# and output folder names (group_1/, group_2/) are untouched -- only text
# that is actually drawn or printed uses these.
GROUP_DISPLAY_NAMES = {"1": "weak ridge", "2": "strong ridge"}


def group_display_name(group):
    """Lowercase display name for a group key, e.g. for use in running text
    ('weak ridge members'). Falls back to 'group {key}' for any key other
    than "1"/"2", so this never silently breaks if a group is renamed/added."""
    return GROUP_DISPLAY_NAMES.get(group, f"group {group}")


def scope_label(group):
    if group == ALL_KEY:
        return "all members"
    return f"{group_display_name(group)} members"
 
 
def legend_handles(k, positions):
    """One legend entry per cluster from the FULL clustering (so the legend is
    identical across groups), annotated with how many of this group's
    trajectories fall in that cluster. n=0 means the group has none there."""
    labels_sel = results[k][positions]
    return [
        Line2D([0], [0], color=cluster_colors[c], linewidth=2,
               label=f"Cluster {c + 1} mean (n={int((labels_sel == c).sum())})")
        for c in np.unique(results[k])
    ]

    

def single_cluster_legend_handles(c):
    """Two legend entries for a single-cluster comparison plot: group 1 (dashed)
    and group 2 (solid), both drawn in that cluster's color, matching the
    color used for this cluster everywhere else in the script."""
    color = cluster_colors[c]
    return [
        Line2D([0], [0], color=color, linestyle="--", linewidth=2,
               label=group_display_name("1").title()),
        Line2D([0], [0], color=color, linestyle="-", linewidth=2,
               label=group_display_name("2").title()),
    ]

def cluster_color_legend_handles(k):
    """One legend entry per cluster, color only (no group/count in the label).
    Used by the all-clusters-combined group comparison plots below, where line
    STYLE (dashed/solid) already distinguishes group 1 from group 2, so the
    color legend only needs to say which cluster each color is."""
    return [
        Line2D([0], [0], color=cluster_colors[c], linewidth=2, label=f"Cluster {c + 1} mean")
        for c in np.unique(results[k])
    ]
 
 
def group_style_legend_handles():
    """Two legend entries (black, color-neutral) explaining the dashed-vs-solid
    convention used by the all-clusters-combined group comparison plots:
    group 1 dashed, group 2 solid."""
    return [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2,
               label=group_display_name("1").title()),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2,
               label=group_display_name("2").title()),
    ]
 
 
def all_clusters_legend_handles(k):
    """Legend for the combined (all-clusters-in-one-figure) comparison map:
    one color swatch per cluster, plus two black, color-neutral entries
    explaining the dashed (group 1) vs solid (group 2) line style."""
    cluster_handles = [
        Line2D([0], [0], color=cluster_colors[c], linewidth=2, label=f"Cluster {c + 1} mean")
        for c in np.unique(results[k])
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2,
               label=group_display_name("1").title()),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2,
               label=group_display_name("2").title()),
    ]
    return cluster_handles + style_handles


# Individual trajectory fundtions - comment out if opting for mean trajectories
 
def cluster_map_plot(k, positions, group, savepath):
    labels = results[k]
    lat, lon = kept("lat"), kept("lon")
    #print(labels)
 
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    ax.set_extent([-90,-150,25,70], crs=ccrs.PlateCarree())
 
    for i in positions:
        if labels[i] == 1:
            ax.plot(lon[:, i], lat[:, i], color=cluster_colors[labels[i]],
                    linewidth=0.5, alpha=0.2, transform=ccrs.PlateCarree())
 
    # ax.legend(handles=legend_handles(k, positions), title="Cluster",
    #           loc="lower center", frameon=True)
    ax.set_title(
        f"Trajectory clusters ({scope_label(group)}, all heights, joint clustering) "
        f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
    )
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
# def cluster_variable_timeseries_plot(k, positions, variable, ylabel, title, savepath,
#                                      xlim=None, legend_loc="upper right"):
#     labels = results[k]
#     vals = kept(variable)
 
#     fig, ax = plt.subplots(figsize=(10, 5))
#     for i in positions:
#         ax.plot(time_vals, vals[:, i], color=cluster_colors[labels[i]],
#                 linewidth=0.5, alpha=0.2)
 
#     ax.legend(handles=legend_handles(k, positions), loc=legend_loc, frameon=True)
#     ax.set_xlabel("Time (hours)")
#     ax.set_ylabel(ylabel)
#     if xlim is not None:
#         ax.set_xlim(*xlim)
#     ax.set_title(title)
#     plt.tight_layout()
#     plt.savefig(savepath, dpi=150, bbox_inches="tight")
#     plt.close(fig)


# Mean plots 
def cluster_mean_spread(vals, labels_sel, c):
    """Mean and spread (per time step) across the trajectories in cluster c.
    vals is (ntim, n_trajectories); labels_sel is the cluster label of each column."""
    sel = vals[:, labels_sel == c]
    n = sel.shape[1]
    mean = np.nanmean(sel, axis=1)
    if n > 1:
        spread = np.nanstd(sel, axis=1, ddof=1)
    else:
        spread = np.zeros_like(mean)     # a single trajectory has no spread
    return mean, spread, n
 
 
def cluster_mean_map_plot(k, positions, group, savepath):
    """Mean track of each cluster"""
    labels_sel = results[k][positions]
    lat = kept("lat")[:, positions]
    lon = kept("lon")[:, positions]
 
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
 
    for c in np.unique(labels_sel):
        mlat, slat, _ = cluster_mean_spread(lat, labels_sel, c)
        mlon, slon, _ = cluster_mean_spread(lon, labels_sel, c)
        color = cluster_colors[c]

        ax.plot(mlon, mlat, color=color, linewidth=2.5, transform=ccrs.PlateCarree())
 
    ax.legend(handles=legend_handles(k, positions),
              loc="lower right", frameon=True)
    ax.set_title(
        f"Mean trajectory of each cluster, "
        f"({scope_label(group)}, all heights, joint clustering) "
        f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
    )
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


TIGHT_YLIM_LABELS = {"PV (PVU) / s", "K / s"}
TIGHT_YLIM_PAD = 0.00001 
 
def tight_ylim(ylabel, *value_arrays):
    """If `ylabel` is one of TIGHT_YLIM_LABELS, return a (min, max) ylim that
    extends TIGHT_YLIM_PAD past the lowest and highest values in
    `value_arrays` (each an array of the raw trajectory values actually drawn
    on that axes). Returns None for any other plot, so the caller's default
    (automatic) y-axis scaling is left alone."""
    if ylabel not in TIGHT_YLIM_LABELS:
        return None
    finite = [v[np.isfinite(v)] for v in value_arrays if v.size]
    finite = [v for v in finite if v.size]
    if not finite:
        return None
    ymin = min(v.min() for v in finite)
    ymax = max(v.max() for v in finite)
    return ymin - TIGHT_YLIM_PAD, ymax + TIGHT_YLIM_PAD
 
 
def cluster_mean_timeseries_plot(k, positions, variable, ylabel, title, savepath,
                                 xlim=None, legend_loc="upper right"):
    """Mean time series of each cluster with a shaded +/- spread band."""
    labels_sel = results[k][positions]
    vals = kept(variable)[:, positions]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted_values = []
    for c in np.unique(labels_sel):
        mean, spread, _ = cluster_mean_spread(vals, labels_sel, c)
        ax.plot(time_vals, mean, color=cluster_colors[c], linewidth=2)
        plotted_values.append(mean)
 
    ax.legend(handles=legend_handles(k, positions), loc=legend_loc, frameon=True)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ylim = tight_ylim(ylabel, *plotted_values)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(f"{title} [mean, std]")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)



# ============================================================================
# 10b. Group 1 vs Group 2 comparison plots (NEW -- in addition to the
#      per-group plots above, not a replacement for them).
# ============================================================================
# One comparison plot PER CLUSTER PER VARIABLE (5 clusters x 12 plot types =
# 60 plots for the default K_TO_PLOT/variable set), each showing only that
# cluster's two group means on one axes: group 1 dashed, group 2 solid, both
# drawn in that cluster's own color (the same color used everywhere else in
# the script).
 
def comparison_map_plot_one_cluster(k, c, savepath):
    pos1, pos2 = group_positions["1"], group_positions["2"]
    labels1, labels2 = results[k][pos1], results[k][pos2]
    lat1, lon1 = kept("lat")[:, pos1], kept("lon")[:, pos1]
    lat2, lon2 = kept("lat")[:, pos2], kept("lon")[:, pos2]
    color = cluster_colors[c]
 
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
 
    mlat1, _, n1 = cluster_mean_spread(lat1, labels1, c)
    mlon1, _, _ = cluster_mean_spread(lon1, labels1, c)
    ax.plot(mlon1, mlat1, color=color, linestyle="--", linewidth=2.5,
            transform=ccrs.PlateCarree())
 
    mlat2, _, n2 = cluster_mean_spread(lat2, labels2, c)
    mlon2, _, _ = cluster_mean_spread(lon2, labels2, c)
    ax.plot(mlon2, mlat2, color=color, linestyle="-", linewidth=2.5,
            transform=ccrs.PlateCarree())
 
    ax.legend(handles=single_cluster_legend_handles(c), loc="lower right", frameon=True)
    ax.set_title(
        f"Cluster {c + 1} mean trajectory, {group_display_name('1')} (dashed, n={n1}) vs "
        f"{group_display_name('2')} (solid, n={n2}) (all heights, joint clustering) "
        f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
    )
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def comparison_timeseries_plot_one_cluster(k, c, variable, ylabel, title, savepath,
                                           xlim=None, legend_loc="upper right"):
    pos1, pos2 = group_positions["1"], group_positions["2"]
    labels1, labels2 = results[k][pos1], results[k][pos2]
    vals1 = kept(variable)[:, pos1]
    vals2 = kept(variable)[:, pos2]
    color = cluster_colors[c]
 
    fig, ax = plt.subplots(figsize=(10, 5))
 
    mean1, spread1, n1 = cluster_mean_spread(vals1, labels1, c)
    ax.plot(time_vals, mean1, color=color, linestyle="--", linewidth=2)
 
    mean2, spread2, n2 = cluster_mean_spread(vals2, labels2, c)
    ax.plot(time_vals, mean2, color=color, linestyle="-", linewidth=2)
 
    ax.legend(handles=single_cluster_legend_handles(c), loc=legend_loc, frameon=True)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ylim = tight_ylim(ylabel, mean1, mean2)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(
        f"Cluster {c + 1}: {title} [mean, std; {group_display_name('1')} dashed n={n1}, "
        f"{group_display_name('2')} solid n={n2}]"
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

 
 
def make_comparison_plots(k, outdir):
    """Every group-1-vs-group-2 comparison plot, for every cluster, for the
    same variable set as make_group_plots below. Each cluster's plots are
    saved in their own subfolder so the 12-plots-per-cluster sets don't mix
    together in one directory listing."""
    suffix = f"{k}_z_weight={Z_WEIGHT}"
    tail = f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
 
    for c in np.unique(results[k]):
        cluster_dir = os.path.join(outdir, f"cluster_{c + 1}")
        os.makedirs(cluster_dir, exist_ok=True)
 
        comparison_map_plot_one_cluster(
            k, c, os.path.join(cluster_dir, f"trajectory_cluster{c + 1}_compare_{suffix}.png"),
        )
 
        comparison_timeseries_plot_one_cluster(
            k, c, "z", "Height (m)",
            f"Height time series {tail}",
            os.path.join(cluster_dir, f"height_time_series_cluster{c + 1}_compare_{suffix}.png"),
            legend_loc="upper left",
        )
 
        comparison_timeseries_plot_one_cluster(
            k, c, "temperature_isobaric", "Temperature (K)",
            f"Temperature time series {tail}",
            os.path.join(cluster_dir, f"temperature_time_series_cluster{c + 1}_compare_{suffix}.png"),
        )
 
        comparison_timeseries_plot_one_cluster(
            k, c, "total_diabatic_tendency_with_friction", "PV (PVU) / s",
            f"Total diabatic PV tendency (with friction) {tail}",
            os.path.join(cluster_dir, f"diabatic_pv_time_series_cluster{c + 1}_with_friction_compare_{suffix}.png"),
            xlim=(0, 48),
        )
 
        comparison_timeseries_plot_one_cluster(
            k, c, "acc_depv_dt_dyn_isobaric", "PV (PVU) / s",
            f"Total dynamic (advection) PV tendency {tail}",
            os.path.join(cluster_dir, f"dynamics_pv_time_series_cluster{c + 1}_compare_{suffix}.png"),
            xlim=(0, 48),
        )

        comparison_timeseries_plot_one_cluster(
            k, c, "acc_dtheta_dt_mp_isobaric", "K / s",
            f"Latent heating rate {tail}",
            os.path.join(cluster_dir, f"latent_heating_rate_time_series_cluster{c + 1}_compare_{suffix}.png"),
            xlim=(0, 48),
        )
 
        if "ivt" in ds:
            comparison_timeseries_plot_one_cluster(
                k, c, "ivt", "IVT (kg m$^{-1}$ s$^{-1}$)",
                f"IVT time series {tail}",
                os.path.join(cluster_dir, f"ivt_time_series_cluster{c + 1}_compare_{suffix}.png"),
                xlim=(0, 48),
            )
 
        for var_name, var_label in tendency_vars.items():
            comparison_timeseries_plot_one_cluster(
                k, c, var_name, "PV (PVU) / s",
                f"{var_label} {tail}",
                os.path.join(cluster_dir, f"{var_name}_cluster{c + 1}_compare_{suffix}.png"),
                xlim=(0, 48),
            )

# ============================================================================
# 10c. Group 1 vs Group 2 comparison plots, ALL CLUSTERS COMBINED (NEW --
#      in addition to both section 10's per-group plots and section 10b's
#      per-cluster comparison plots above; nothing above is replaced).
# ============================================================================
# One combined plot PER VARIABLE (not per cluster): this takes
# cluster_mean_map_plot's / cluster_mean_timeseries_plot's separate group 1
# and group 2 figures (each already showing every cluster's mean) and
# overlays them into a single figure -- every cluster's mean is drawn twice
# on the same axes, group 1 dashed and group 2 solid, each in that cluster's
# own color. The map plot is mean-only (no spread), matching
# cluster_mean_map_plot.
 
def all_clusters_comparison_map_plot(k, savepath):
    pos1, pos2 = group_positions["1"], group_positions["2"]
    labels1, labels2 = results[k][pos1], results[k][pos2]
    lat1, lon1 = kept("lat")[:, pos1], kept("lon")[:, pos1]
    lat2, lon2 = kept("lat")[:, pos2], kept("lon")[:, pos2]
 
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.LAKES, facecolor="white")
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
 
    for c in np.unique(results[k]):
        color = cluster_colors[c]
        if c in labels1:
            mlat, _, _ = cluster_mean_spread(lat1, labels1, c)
            mlon, _, _ = cluster_mean_spread(lon1, labels1, c)
            ax.plot(mlon, mlat, color=color, linestyle="--", linewidth=2.5,
                    transform=ccrs.PlateCarree())
        if c in labels2:
            mlat, _, _ = cluster_mean_spread(lat2, labels2, c)
            mlon, _, _ = cluster_mean_spread(lon2, labels2, c)
            ax.plot(mlon, mlat, color=color, linestyle="-", linewidth=2.5,
                    transform=ccrs.PlateCarree())
 
    ax.legend(handles=cluster_color_legend_handles(k) + group_style_legend_handles(),
              loc="lower center", frameon=True)
    ax.set_title(
        f"Mean trajectory by cluster, all clusters combined -- {group_display_name('1')} (dashed) "
        f"vs {group_display_name('2')} (solid) (all heights, joint clustering) "
        f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
    )
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    

AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 10
POSTER_LEGEND_FONTSIZE = 18
 
def all_clusters_comparison_timeseries_plot(k, variable, ylabel, title, savepath,
                                            xlim=None, legend_loc="upper right",
                                            show_legend=True, legend_fontsize=None):
    pos1, pos2 = group_positions["1"], group_positions["2"]
    labels1, labels2 = results[k][pos1], results[k][pos2]
    vals1 = kept(variable)[:, pos1]
    vals2 = kept(variable)[:, pos2]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted_values = []
    for c in np.unique(results[k]):
        color = cluster_colors[c]
        if c in labels1:
            mean, spread, _ = cluster_mean_spread(vals1, labels1, c)
            ax.plot(time_vals, mean, color=color, linestyle="--", linewidth=2)
            plotted_values.append(mean)
        if c in labels2:
            mean, spread, _ = cluster_mean_spread(vals2, labels2, c)
            ax.plot(time_vals, mean, color=color, linestyle="-", linewidth=2)
            plotted_values.append(mean)
 
    if show_legend:
        ax.legend(handles=cluster_color_legend_handles(k) + group_style_legend_handles(),
                  loc=legend_loc, frameon=True,
                  fontsize=legend_fontsize if legend_fontsize is not None else LEGEND_FONTSIZE)
    ax.set_xlabel("Time (hours)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ylim = tight_ylim(ylabel, *plotted_values)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(
        f"{title}, all clusters combined [mean, std; "
        f"{group_display_name('1')} dashed, {group_display_name('2')} solid]", pad = 20
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def make_all_clusters_comparison_plots(k, outdir, legend_variable=None):
    """...
    legend_variable: if None (default), every plot keeps its normal-size
    legend -- unchanged behavior. If set to one of the variable name strings
    below (e.g. "acc_depv_dt_dyn_isobaric"), ONLY that plot gets a legend, and
    it's drawn at POSTER_LEGEND_FONTSIZE; every other plot's legend is
    suppressed entirely. ..."""
    suffix = f"{k}_z_weight={Z_WEIGHT}"
    tail = f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"

    def legend_kwargs(var_name):
        if legend_variable is None:
            return dict(show_legend=True, legend_fontsize=None)
        is_legend_plot = (var_name == legend_variable)
        return dict(show_legend=is_legend_plot,
                    legend_fontsize=POSTER_LEGEND_FONTSIZE if is_legend_plot else None)
 
    all_clusters_comparison_map_plot(
        k, os.path.join(outdir, f"trajectory_clusters_all_heights_compare_{suffix}.png"),
    )
 
    all_clusters_comparison_timeseries_plot(
        k, "z", "Height (m)",
        f"Height time series by cluster {tail}",
        os.path.join(outdir, f"height_time_series_clusters_all_heights_compare_{suffix}.png"),
        xlim=(0, 36),
        legend_loc="upper left",
        **legend_kwargs("z"), 
    )
 
    all_clusters_comparison_timeseries_plot(
        k, "temperature_isobaric", "Temperature (K)",
        f"Temperature time series by cluster {tail}",
        os.path.join(outdir, f"temperature_time_series_clusters_compare_{suffix}.png"),
        xlim=(0, 36),
        **legend_kwargs("temperature_isobaric"),
    )
 
    all_clusters_comparison_timeseries_plot(
        k, "total_diabatic_tendency_with_friction", "PV (PVU) / s",
        f"Total diabatic PV tendency (with friction) by cluster {tail}",
        os.path.join(outdir, f"diabatic_pv_time_series_clusters_with_friction_compare_{suffix}.png"),
        xlim=(0, 36),
        **legend_kwargs("total_diabatic_tendency_with_friction"),
    )
 
    all_clusters_comparison_timeseries_plot(
        k, "acc_depv_dt_dyn_isobaric", "PV (PVU) / s",
        f"Total dynamic (advection) PV tendency by cluster {tail}",
        os.path.join(outdir, f"dynamics_pv_time_series_clusters_compare_{suffix}.png"),
        xlim=(0, 36),
        **legend_kwargs("acc_depv_dt_dyn_isobaric"),
    )

    all_clusters_comparison_timeseries_plot(
            k, "acc_dtheta_dt_mp_isobaric", "K / s",
            f"Latent heating rate {tail}",
            os.path.join(outdir, f"latent_heating_rate_time_series_clusters_compare_{suffix}.png"),
            xlim=(0, 36),
        **legend_kwargs("acc_dtheta_dt_mp_isobaric"),
     )
 
    if "ivt" in ds:
        all_clusters_comparison_timeseries_plot(
            k, "ivt", "IVT (kg m$^{-1}$ s$^{-1}$)",
            f"IVT time series by cluster {tail}",
            os.path.join(outdir, f"ivt_time_series_clusters_compare_{suffix}.png"),
            xlim=(0, 36),
            **legend_kwargs("ivt"),
        )
 
    for var_name, var_label in tendency_vars.items():
        all_clusters_comparison_timeseries_plot(
            k, var_name, "PV (PVU) / s",
            f"{var_label} by cluster {tail}",
            os.path.join(outdir, f"{var_name}_clusters_compare_{suffix}.png"),
            xlim=(0, 36),
            **legend_kwargs(f"{var_name}"),
        )

 
 
def write_cluster_counts(k, group_positions, savepath):
    """Cluster x group table (count and % of that group's trajectories)."""
    labels = results[k]
    clusters = np.unique(labels)
    names = list(group_positions)
 
    def col_name(g):
        return "All members" if g == ALL_KEY else group_display_name(g).title()
 
    lines = [
        f"Cluster counts for k={k}: clustered jointly on {len(labels)} trajectories "
        f"from {n_members} members (after domain filter).",
        "",
    ]
    for g in names:
        mem = "" if g == ALL_KEY else f"  (members {sorted(groups[g])})"
        lines.append(f"{col_name(g)}: {len(group_positions[g])} trajectories{mem}")
    lines.append("")
 
    lines.append(f"{'':<11}" + "".join(f"{col_name(g):>18}" for g in names))
    for c in clusters:
        row = f"{'Cluster ' + str(c + 1):<11}"
        for g in names:
            pos = group_positions[g]
            n = int((labels[pos] == c).sum())
            pct = 100 * n / len(pos) if len(pos) else 0.0
            row += f"{f'{n} ({pct:.1f}%)':>18}"
        lines.append(row)
 
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(savepath, "w") as fh:
        fh.write(summary + "\n")
 
 
# Individual PV tendency components (same set as the single-member script).
tendency_vars = {
    "acc_depv_dt_lw_isobaric": "Longwave radiation",
    "acc_depv_dt_sw_isobaric": "Shortwave radiation",
    "acc_depv_dt_bl_isobaric": "Boundary layer scheme",
    "acc_depv_dt_cu_isobaric": "Cumulus scheme",
    "acc_depv_dt_mp_isobaric": "Microphysics",
    "acc_depv_dt_fric_bl_isobaric": "Boundary layer friction",
}
 
 
def make_group_plots(k, positions, group, outdir):
    """Every cluster plot for one group, using the shared joint-clustering labels."""
    scope = scope_label(group)
    tail = f"initialized at {formatted_time} for {k} clusters, height weight of {Z_WEIGHT}"
    suffix = f"{k}_z_weight={Z_WEIGHT}"

    # ----- Individual-trajectory plots -----
    cluster_map_plot(
        k, positions, group,
        os.path.join(outdir, f"trajectory_clusters_all_heights_{suffix}.png"),
    )
 
    # cluster_variable_timeseries_plot(
    #     k, positions, "z", "Height (m)",
    #     f"Height time series by cluster ({scope}) {tail}",
    #     os.path.join(outdir, f"height_time_series_clusters_all_heights_{suffix}.png"),
    #     legend_loc="upper left",
    # )
 
    # cluster_variable_timeseries_plot(
    #     k, positions, "temperature_isobaric", "Temperature (K)",
    #     f"Temperature time series by cluster ({scope}) {tail}",
    #     os.path.join(outdir, f"temperature_time_series_clusters_{suffix}.png"),
    # )
 
    # cluster_variable_timeseries_plot(
    #     k, positions, "total_diabatic_tendency_with_friction", "PV (PVU) / s",
    #     f"Total diabatic PV tendency (with friction) by cluster ({scope}) {tail}",
    #     os.path.join(outdir, f"diabatic_pv_time_series_clusters_with_friction_{suffix}.png"),
    #     xlim=(0, 20),
    # )
 
    # cluster_variable_timeseries_plot(
    #     k, positions, "acc_depv_dt_dyn_isobaric", "PV (PVU) / s",
    #     f"Total dynamic (advection) PV tendency by cluster ({scope}) {tail}",
    #     os.path.join(outdir, f"dynamics_pv_time_series_clusters_{suffix}.png"),
    #     xlim=(0, 20),
    # )
 
    # if "ivt" in ds:
    #     cluster_variable_timeseries_plot(
    #         k, positions, "ivt", "IVT (kg m$^{-1}$ s$^{-1}$)",
    #         f"IVT time series by cluster ({scope}) {tail}",
    #         os.path.join(outdir, f"ivt_time_series_clusters_{suffix}.png"),
    #         xlim=(0, 20),
    #     )
 
    # for var_name, var_label in tendency_vars.items():
    #     cluster_variable_timeseries_plot(
    #         k, positions, var_name, "PV (PVU) / s",
    #         f"{var_label} by cluster ({scope}) {tail}",
    #         os.path.join(outdir, f"{var_name}_clusters_{suffix}.png"),
    #         xlim=(0, 20),
    #     )

    m = f"mean_std"
 
    cluster_mean_map_plot(
        k, positions, group,
        os.path.join(outdir, f"trajectory_clusters_all_heights_{m}_{suffix}.png"),
    )
 
    cluster_mean_timeseries_plot(
        k, positions, "z", "Height (m)",
        f"Height time series by cluster ({scope}) {tail}",
        os.path.join(outdir, f"height_time_series_clusters_all_heights_{m}_{suffix}.png"),
        xlim=(0, 48),
        legend_loc="upper left",
    )
 
    cluster_mean_timeseries_plot(
        k, positions, "temperature_isobaric", "Temperature (K)",
        f"Temperature time series by cluster ({scope}) {tail}",
        os.path.join(outdir, f"temperature_time_series_clusters_{m}_{suffix}.png"),
        xlim=(0, 48),
    )
 
    cluster_mean_timeseries_plot(
        k, positions, "total_diabatic_tendency_with_friction", "PV (PVU) / s",
        f"Total diabatic PV tendency (with friction) by cluster ({scope}) {tail}",
        os.path.join(outdir, f"diabatic_pv_time_series_clusters_with_friction_{m}_{suffix}.png"),
        xlim=(0, 48),
    )
 
    cluster_mean_timeseries_plot(
        k, positions, "acc_depv_dt_dyn_isobaric", "PV (PVU) / s",
        f"Total dynamic (advection) PV tendency by cluster ({scope}) {tail}",
        os.path.join(outdir, f"dynamics_pv_time_series_clusters_{m}_{suffix}.png"),
        xlim=(0, 48),
    )

    cluster_mean_timeseries_plot(
        k, positions, "acc_dtheta_dt_mp_isobaric", "K / s",
        f"Latent heating rate by cluster ({scope}) {tail}",
        os.path.join(outdir, f"latent_heating_rate_time_series_clusters_{m}_{suffix}.png"),
        xlim=(0, 48),
    )
 
    if "ivt" in ds:
        cluster_mean_timeseries_plot(
            k, positions, "ivt", "IVT (kg m$^{-1}$ s$^{-1}$)",
            f"IVT time series by cluster ({scope}) {tail}",
            os.path.join(outdir, f"ivt_time_series_clusters_{m}_{suffix}.png"),
            xlim=(0, 48),
        )
 
    for var_name, var_label in tendency_vars.items():
        cluster_mean_timeseries_plot(
            k, positions, var_name, "PV (PVU) / s",
            f"{var_label} by cluster ({scope}) {tail}",
            os.path.join(outdir, f"{var_name}_clusters_{m}_{suffix}.png"),
            xlim=(0, 48),
        )
 
 
for k in K_TO_PLOT:
    print(f"\nGenerating cluster plots for k={k} ...")
 
    write_cluster_counts(
        k, group_positions,
        os.path.join(OUTPUT_DIR, f"cluster_counts_{k}_z_weight={Z_WEIGHT}.txt"),
    )
 
    for group, positions in group_positions.items():
        if len(positions) == 0:
            print(f"WARNING: group '{group}' has no trajectories left after the domain "
                  "filter -- skipping its plots.")
            continue
        group_dir = os.path.join(OUTPUT_DIR, group if group == ALL_KEY else f"group_{group}")
        os.makedirs(group_dir, exist_ok=True)
        print(f"  plotting {scope_label(group)} ({len(positions)} trajectories) -> {group_dir}")
        make_group_plots(k, positions, group, group_dir)

    # ----- Group 1 vs Group 2 comparison plots (NEW, in addition to the -----
    # ----- per-group plots above) -----
    if "1" in group_positions and "2" in group_positions:
        if len(group_positions["1"]) == 0 or len(group_positions["2"]) == 0:
            print("WARNING: group '1' or group '2' has no trajectories left after "
                  "the domain filter -- skipping the group-comparison plots.")
        else:
            comparison_dir = os.path.join(OUTPUT_DIR, "group_comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            print(f"  plotting group 1 vs group 2 comparison -> {comparison_dir}")
            make_comparison_plots(k, comparison_dir)

            all_clusters_comparison_dir = os.path.join(OUTPUT_DIR, "group_comparison_all_clusters")
            os.makedirs(all_clusters_comparison_dir, exist_ok=True)
            print(f"  plotting group 1 vs group 2 comparison (all clusters combined) -> {all_clusters_comparison_dir}")
            make_all_clusters_comparison_plots(k, all_clusters_comparison_dir,
                                  legend_variable='total_diabatic_tendency_with_friction')

    else:
        print("NOTE: groups '1' and '2' must both be defined to make comparison "
              "plots -- skipping.")

 
print(f"\nDone. Clustered {len(all_indices)} trajectories from {n_members} member(s) jointly; "
      f"plots saved under: {os.path.abspath(OUTPUT_DIR)}")