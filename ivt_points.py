import os, sys, glob
import datetime as dt
import argparse
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd

sys.path.append('/tornlab_rit/torn/MPAS/python')
from mpas_tools import background_map, regular_interpolate, mpas_to_latlon

exp_parser = argparse.ArgumentParser()
exp_parser.add_argument('--member', action='store', type=str, required=True)
args = exp_parser.parse_args()

if __name__ == '__main__':

  fcst_str = '2023010400'
  ivtmin = 1000
  lat1 = 25
  lon1 = -150
  lat2 = 40
  lon2 = -130

  levlist = [500, 1000, 1500, 2000, 2500]

  fcst_dt = dt.datetime.strptime(fcst_str, '%Y%m%d%H')

  ncfile = Dataset('member9/init.nc','r')
  lon_plt, lat_plt = regular_interpolate(dx=0.5, lon_min=lon1, lat_min=lat1, lon_max=lon2, lat_max=lat2)
  cell_coords = np.vstack((ncfile.variables['lonCell'][:],ncfile.variables['latCell'][:])) * (180./np.pi)
  cell_coords[0] = (cell_coords[0] + 180.) % 360. - 180.
  cellarea = ncfile.variables['areaCell'][:]
  ncfile.close()

  ds = Dataset('member{0}/diag.{1}.nc'.format(args.member, fcst_dt.strftime("%Y-%m-%d_%H.%M.00")),'r')
  ivt = mpas_to_latlon(np.squeeze(ds.variables['ivt'][0,:]), cell_coords, lat_plt, lon_plt)
  nlat = len(ivt[:,0])
  nlon = len(ivt[0,:])
  
  maxloc = np.where(ivt == ivt.max())
  icen   = int(maxloc[1].item())
  jcen   = int(maxloc[0].item())

  fmgrid = ivt.copy(deep=True)
  fmgrid[:,:] = 0.0
  fmgrid[jcen,icen] = 1.0

  iloc       = np.zeros(nlon*nlat, dtype=int)
  jloc       = np.zeros(nlon*nlat, dtype=int)
  nloc       = 0
  iloc[nloc] = icen
  jloc[nloc] = jcen

  k = 0
  while k <= nloc:

    for i in range(max(iloc[k]-1,0), min(iloc[k]+2,nlon)):
      for j in range(max(jloc[k]-1,0), min(jloc[k]+2,nlat)):
        if ivt[j,i] >= ivtmin and fmgrid[j,i] < 1.0:
          nloc = nloc + 1
          iloc[nloc] = i
          jloc[nloc] = j
          fmgrid[j,i] = 1.0

    k = k + 1

  for k in range(nloc):
    for i in range(len(levlist)):

      if 'dfout' in locals():
        dfout = pd.concat([dfout, pd.DataFrame({'hour': [24], 'longitude': [ivt.longitude.values[iloc[k]]], 'latitude': [ivt.latitude.values[jloc[k]]], 'level': [levlist[i]]})], ignore_index = True)
      else:
        dfout = pd.DataFrame({'hour': [24], 'longitude': [ivt.longitude.values[iloc[k]]], 'latitude': [ivt.latitude.values[jloc[k]]], 'level': [levlist[i]]})

  dfout.to_csv('traj_start_{0}'.format(args.member), header=False, index=False, sep='\t')
