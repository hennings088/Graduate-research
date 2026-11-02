############
# same as lagranto_mpas_prep.py, just makes a single file
############

import os, shutil
import numpy as np
import netCDF4 as nc
import argparse
import datetime as dt

convertdir = '/network/rit/lab/tornlab_rit/torn/MPAS/convert_mpas'
initfile   = '/network/rit/lab/tornlab_rit/hs559346/member9/init.nc'

#  Read the file information from the command line
exp_parser = argparse.ArgumentParser()
exp_parser.add_argument('--diagfile', action='store', type=str, required=True)
exp_parser.add_argument('--outdir',   action='store', type=str, default=os.getcwd())
args = exp_parser.parse_args()

diagfile = os.path.abspath(args.diagfile)

#  convert_mpas writes latlon.nc into the current directory, so work in outdir
os.makedirs(args.outdir, exist_ok=True)
os.chdir(args.outdir)

diaglist = [diagfile]
#print(diaglist)

#  time string from the file name, e.g. diag.2023-01-07_00.00.00.nc -> 2023-01-07_00:00:00
datestr = '.'.join(os.path.basename(diagfile).split('.')[1:4])
stamp   = dt.datetime.strptime(datestr, '%Y-%m-%d_%H.%M.%S').strftime('%Y-%m-%d_%H:%M:%S')
outname = 'latlon.current.{0}.nc'.format(stamp)

#  convert the MPAS diagnostic file into lat/lon grid
os.system('{0}/convert_mpas {1} {2}'.format(convertdir, initfile, ' '.join(diaglist)))
shutil.move('latlon.nc', outname)

ncfile = nc.Dataset(outname, mode='a')

#  Add new dimension for MPAS dates
str_dim = ncfile.createDimension('StrLen', 64)

#  Create a xtime variable to store valid times
time_out = ncfile.createVariable('xtime','S1',('Time','StrLen'))
time_out.units = 'YYYY-MM-DD_hh:mm:ss'
time_out.long_name = 'Model valid time'

#  Create a variable to store the pressure levels
iso_out = ncfile.createVariable('iso_levels',np.float32,('nIsoLevels'))
iso_out.units = 'Pa'
iso_out.long_name = 'Levels for vertical interpolation to isobaric surfaces'

pres_out = ncfile.createVariable('pressure',np.float32,('Time','nIsoLevels','latitude','longitude'))
pres_out.units = 'Pa'
pres_out.long_name = 'pressure of isobaric surfaces'

ncdiag = nc.Dataset(diaglist[0], mode='r')
iso_out[:] = ncdiag.variables['iso_levels'][:]
ncdiag.close()

#  Write the time from the diagnostic file into the lat/lon file
for i in range(len(diaglist)):

  ncdiag = nc.Dataset(diaglist[i], mode='r')
  time_out[i,:] = ncdiag.variables['xtime'][0,:]
  ncdiag.close()

ncdiag = nc.Dataset(diaglist[0], mode='r')
for k in range(len(ncdiag.variables['iso_levels'][:])):
  pres_out[:,k,:,:] = ncdiag.variables['iso_levels'][k]
ncdiag.close()

if not 'ter' in ncdiag.variables:

  os.system('{0}/convert_mpas {1}'.format(convertdir,initfile))

  ter_out = ncfile.createVariable('ter',np.float32,('latitude','longitude'))
  ter_out.units = 'm'
  ter_out.long_name = 'terrain height'

  ncinit = nc.Dataset('latlon.nc', mode='r')
  ter_out[:,:] = ncinit.variables['ter'][:,:]
  ncinit.close()

  os.remove('latlon.nc')

ncfile.close()

print('Wrote {0}/{1}'.format(args.outdir, outname))
