############
# Deprecated in favor of ivt_points.py which creates a dynamic set of starting points based on ivt magnitude
############

import numpy as np

# ---- User settings ----
center_lon = -138
center_lat = 32

# horizontal spacing in degrees
dlon = 0.25
dlat = 0.25

# vertical levels (meters)
heights = [50, 100, 150, 200, 250]

# relative time
time0 = 24.00
# -----------------------

# Create 3x3 grid
lons = np.arange(center_lon - 1.0, center_lon + 1.0 + dlon/2, dlon)
lats = np.arange(center_lat - 1.0, center_lat + 1.0 + dlat/2, dlat)

with open("startf_z_lower", "w") as f:
    for lon in lons:
        for lat in lats:
            for z in heights:
                f.write(f"{time0:6.2f} {lon:8.2f} {lat:8.2f} {z:10.2f}\n")

