############
# creates a comparison panel comparing PV across sets of ensemble members at multiple times
############


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from datetime import datetime

# Define the two groups
group1 = [4, 12, 36, 37, 48, 66, 72, 87, 90]
group2 = [2, 23, 26, 33, 42, 47, 50, 61, 83]

basedir = '/nfs/tornlab_rit/hs559346'
pv_level = 8

##### -------------------- #####
top_dates = ['2023-01-04', '2023-01-05', '2023-01-06']   # top row, left to right
an_date   = '2023-01-07'                                 # bottom row (and analysis time)
##### -------------------- #####

##### -------------------- #####
start_west = -150        # western edge of the map on the first date (eastern N. Pacific)
##### -------------------- #####

extent = [-120, -50, 20, 70]                 # final extent (unchanged)
box_w  = extent[1] - extent[0]               # 70 deg wide
end_west = extent[0]

# Weak - Strong difference dates, in order: 4th, 5th, 6th, 7th
shift_dates = top_dates + [an_date]
west_edges = np.linspace(start_west, end_west, len(shift_dates))
shift_extent = {d: [w, w + box_w, extent[2], extent[3]]
                for d, w in zip(shift_dates, west_edges)}

def fname(date):
    return f'latlon.current.{date}_00:00:00.nc'

def pretty(date):
    """'2023-01-04' -> '4 January'"""
    d = datetime.strptime(date, '%Y-%m-%d')
    return f'{d.day} {d.strftime("%B")}'

def load_pv(members, basedir, latlon_file, pv_level):
    """Load and average PV at a given level across all members in a group."""
    pv_list = []
    for member in members:
        filepath = f'{basedir}/member{member}_lagranto/{latlon_file}'
        try:
            ds = xr.open_dataset(filepath)
            pv = ds['ertel_pv_isobaric'].isel(Time=0, nIsoLevels=pv_level).values
            pv_list.append(pv)
            ds.close()
            print(f"  Member {member} loaded successfully")
        except FileNotFoundError:
            print(f"  File not found for member {member}, skipping")
        except Exception as e:
            print(f"  Error loading member {member}: {e}")
    return np.mean(pv_list, axis=0)

# Load lat/lon grid from one of the latlon files
ds_ref = xr.open_dataset(f'{basedir}/member2_lagranto/{fname(an_date)}')
lat_grid = ds_ref['latitude'].values
lon_grid = ds_ref['longitude'].values
ds_ref.close()

# ---- Group differences (Weak - Strong) and ensemble means for each date ----
group_diff = {}   # date -> Weak ridge - Strong ridge
ens_mean   = {}   # date -> mean of both groups (used for contours)
for date in top_dates + [an_date]:
    print(f"\n=== {date} ===")
    print("Loading group 1...")
    g1 = load_pv(group1, basedir, fname(date), pv_level)
    print("Loading group 2...")
    g2 = load_pv(group2, basedir, fname(date), pv_level)
    group_diff[date] = g1 - g2
    ens_mean[date]   = (g1 + g2) / 2
    if date == an_date:
        pv_group1, pv_group2 = g1, g2

# ---- Analysis on an_date ----
ds_rean = xr.open_dataset(f'{basedir}/{fname(an_date)}')
pv_rean = ds_rean['ertel_pv_isobaric'].isel(Time=0, nIsoLevels=pv_level).values
lat_rean = ds_rean['latitude'].values
lon_rean = ds_rean['longitude'].values
ds_rean.close()

pv_rean_smooth = gaussian_filter(pv_rean, sigma=6)

rean_da = xr.DataArray(pv_rean_smooth,
                       coords={'latitude': lat_rean, 'longitude': lon_rean},
                       dims=['latitude', 'longitude'])
pv_rean_ens = rean_da.interp(latitude=lat_grid, longitude=lon_grid).values

pv_diff_r1 = pv_group1 - pv_rean_ens   # Weak ridge - Analysis
pv_diff_r2 = pv_group2 - pv_rean_ens   # Strong ridge - Analysis

# ---- One shared, symmetric scale for all 6 panels ----
all_fields = list(group_diff.values()) + [pv_diff_r1, pv_diff_r2]
dmax = max(np.nanmax(np.abs(f)) for f in all_fields)
diff_levels = np.linspace(-dmax, dmax, 21)

def int_levels(field):
    return np.arange(np.floor(field.min()), np.ceil(field.max()) + 1, 1)

# ---- Panel definitions: (row, col, data, title, contour field, contour levels) ----
panels = []
for col, date in enumerate(top_dates):
    panels.append((0, col, group_diff[date],
                   f'Weak ridge - Strong ridge 300 hPa PV, {pretty(date)}',
                   ens_mean[date], int_levels(ens_mean[date]),
                   shift_extent[date]))
panels += [
    (1, 0, group_diff[an_date],
     f'Weak ridge - Strong ridge 300 hPa PV, {pretty(an_date)}',
     ens_mean[an_date], int_levels(ens_mean[an_date]),
     shift_extent[an_date]),
    (1, 1, pv_diff_r1, f'Weak ridge - Analysis 300 hPa PV, {pretty(an_date)}', None, None, extent),
    (1, 2, pv_diff_r2, f'Strong ridge - Analysis 300 hPa PV, {pretty(an_date)}', None, None, extent),
]

# ---- Figure geometry (inches), so the maps pack with almost no white space ----
#extent = [-120, -50, 20, 70]
map_w   = 6.0
map_h   = map_w * (extent[3] - extent[2]) / (extent[1] - extent[0])  # PlateCarree aspect
title_h = 0.25   # room above each map for its title (this is the only gap between rows)
gap_x   = 0.1   # horizontal gap between maps
left    = 0.05
cb_pad, cb_h, cb_room = 0.1, 0.22, 0.7   # gap below maps, bar thickness, room for ticks + label

maps_w = 3 * map_w + 2 * gap_x           # total width of the map block
fig_w = left + maps_w + 0.1              # no right-hand colorbar column anymore
fig_h = 2 * (map_h + title_h) + cb_pad + cb_h + cb_room

fig = plt.figure(figsize=(fig_w, fig_h))
proj = ccrs.PlateCarree(central_longitude=180)

bottom = cb_pad + cb_h + cb_room         # space reserved at the bottom

def add_map(row, col):
    x = left + col * (map_w + gap_x)
    y = fig_h - (row + 1) * (map_h + title_h)
    return fig.add_axes([x / fig_w, y / fig_h, map_w / fig_w, map_h / fig_h],
                        projection=proj)

for row, col, dat, title, cfield, clevels, ext in panels:
    ax = add_map(row, col)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_extent(ext, crs=ccrs.PlateCarree())

    cf = ax.contourf(lon_grid, lat_grid, dat, levels=diff_levels,
                     cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())

    if cfield is not None:
        cs = ax.contour(lon_grid, lat_grid, cfield, levels=clevels, colors='k',
                        linewidths=1.2, transform=ccrs.PlateCarree())
        ax.clabel(cs, levels=clevels, inline=True, fontsize=8, fmt='%d')

    ax.set_title(title, fontsize=14)

# ---- Single horizontal colorbar below both rows, centered ----
cb_len = 0.6 * maps_w                                   # bar length
cb_x   = left + (maps_w - cb_len) / 2                   # centered under the maps
cb_y   = cb_room                                        # bottom of the bar, above the label space
cax = fig.add_axes([cb_x / fig_w, cb_y / fig_h, cb_len / fig_w, cb_h / fig_h])
cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')
cbar.set_label('Ertel PV difference (PVU)', fontsize=11)

plt.savefig('pv_300hpa_group_comparison_20230104-0107.png', dpi=150, bbox_inches='tight')
plt.show()