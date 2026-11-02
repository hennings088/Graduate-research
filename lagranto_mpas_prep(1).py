import os, glob, shutil
import numpy as np
import xarray as xr
import netCDF4 as nc
import argparse
import datetime as dt

convertdir = '/network/rit/lab/tornlab_rit/torn/MPAS/convert_mpas'
initfile   = '/network/rit/lab/tornlab_rit/hs559346/member9/init.nc'
basedir = '/nfs/tornlab_rit/hs559346'
instant    = False
dtpv       = 1

#  Read the forecast information from the command line
exp_parser = argparse.ArgumentParser()
exp_parser.add_argument('--init',  action='store', type=str, required=True)
exp_parser.add_argument('--fhrmax', action='store', type=float)
exp_parser.add_argument('--member', action='store', type=str, required=True)
args = exp_parser.parse_args()

init_dt = dt.datetime.strptime(args.init, '%Y%m%d%H')
end_dt  = init_dt + dt.timedelta(hours=args.fhrmax)

lalolist = []

curr_dt = init_dt
while curr_dt <= end_dt:

  yyyymmdd = curr_dt.strftime("%Y-%m-%d")

  #  create a list of all diagnostic files
  diaglist = glob.glob('{0}/member{1}/diag.{2}*.nc'.format(basedir, args.member, yyyymmdd))
  print(diaglist)

  ncfile = nc.Dataset(diaglist[0])
  dtmod = ncfile.getncattr('config_dt')
  ncfile.close()

  #  convert the MPAS diagnostic files into lat/lon grid
  os.system('{0}/convert_mpas {1} {2}'.format(convertdir, initfile, ' '.join(diaglist)))
  shutil.move('latlon.nc','latlon_diag.nc')

  pvlist = [item.replace("diag", "pvbudget") for item in diaglist]

  if os.path.isfile(pvlist[0] and instant):

    os.system('{0}/convert_mpas {1} {2}'.format(convertdir,initfile, ' '.join(pvlist)))
    shutil.move('latlon.nc','latlon_pvbudget.nc')

    ds1 = xr.open_dataset('latlon_diag.nc')
    ds2 = xr.open_dataset('latlon_pvbudget.nc')
    combined = xr.merge([ds1, ds2])
    combined.to_netcdf('latlon.current.{0}_00:00:00.nc'.format(yyyymmdd))
#    os.system('ncrcat -h latlon_diag.nc latlon_pvbudget.nc -O latlon.current.{0}_00:00:00.nc'.format(yyyymmdd))
    os.remove('latlon_diag.nc')
    os.remove('latlon_pvbudget.nc')

    lalolist.append('latlon.current.{0}_00:00:00.nc'.format(yyyymmdd))

  elif os.path.isfile(pvlist[0]) and (not instant):

    datelist = ['{0}.{1}.{2}'.format(item.split(".")[1],item.split(".")[2],item.split(".")[3]) for item in diaglist]
    pvlist1  = []
    pvlist2  = []
    dtime    = []
    for datestr in datelist:

      date_dt = dt.datetime.strptime(datestr, '%Y-%m-%d_%H.%M.%S')

      dt1 = date_dt + dt.timedelta(hours=-dtpv)
      if not os.path.isfile('{0}/member{1}/pvbudget.{2}.nc'.format(basedir, args.member, dt1.strftime("%Y-%m-%d_%H.%M.%S"))):
        dt1 = date_dt
      pvlist1.append('{0}/member{1}/pvbudget.{2}.nc'.format(basedir, args.member, dt1.strftime("%Y-%m-%d_%H.%M.%S")))

      dt2 = date_dt + dt.timedelta(hours=dtpv)
      if not os.path.isfile('{0}/member{1}/pvbudget.{2}.nc'.format(basedir, args.member, dt2.strftime("%Y-%m-%d_%H.%M.%S"))):
        dt2 = date_dt
      pvlist2.append('{0}/member{1}/pvbudget.{2}.nc'.format(basedir, args.member, dt2.strftime("%Y-%m-%d_%H.%M.%S")))

      date_dt = dt2-dt1
      dtime.append(date_dt.total_seconds())


    os.system('{0}/convert_mpas {1} {2} '.format(convertdir,initfile, ' '.join(pvlist1)))
    shutil.move('latlon.nc','latlon_pvbudget1.nc')

    os.system('{0}/convert_mpas {1} {2} '.format(convertdir,initfile, ' '.join(pvlist2)))
    shutil.move('latlon.nc','latlon_pvbudget2.nc')

    ncid2 = nc.Dataset('latlon_pvbudget2.nc', 'a')
    var_names = [item for item in list(ncid2.variables.keys()) if 'acc' in item]
    ntime = len(ncid2.dimensions['Time'])

    ncid1 = nc.Dataset('latlon_pvbudget1.nc', 'r')

    for varble in var_names:
      for itime in range(ntime):
        ncid2[varble][itime,:] = (ncid2[varble][itime,:] - ncid1[varble][itime,:]) * dtmod / dtime[itime]

    ncid1.close()
    ncid2.close()

    ds1 = xr.open_dataset('latlon_diag.nc')
    ds2 = xr.open_dataset('latlon_pvbudget2.nc')
    combined = xr.merge([ds1, ds2])
    combined.to_netcdf('latlon.current.{0}_00:00:00.nc'.format(yyyymmdd))
    os.remove('latlon_diag.nc')
    os.remove('latlon_pvbudget1.nc')
    os.remove('latlon_pvbudget2.nc')

  else:

    shutil.move('latlon.nc','latlon.current.{0}_00:00:00.nc'.format(yyyymmdd))


  ncfile = nc.Dataset('latlon.current.{0}_00:00:00.nc'.format(yyyymmdd), mode='a')

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

  #  Write the time from each diagnostic file into the lat/lon file
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

  curr_dt = curr_dt + dt.timedelta(hours=24)


if len(lalolist) > 0:

  ncid = nc.Dataset(lalolist[0], 'r')
  var_names = [item for item in list(ncid.variables.keys()) if 'acc' in item]
#  for varble in var_names:
#    if ncid.variables[varble].ndim < 4:
#      var_names.remove(varble)
#  print(var_names)
  ncid.close()

  for ifile in range(len(lalolist)-1,-1,-1):

    ncid = nc.Dataset(lalolist[ifile], 'a')
    ntime = len(ncid.dimensions['Time'])

    for itime in range(ntime-1,0,-1):

      print(nc.chartostring(ncid.variables['xtime'][itime,:]))
      for varble in var_names:
        ncid[varble][itime,:] = ncid[varble][itime,:] - ncid[varble][itime-1,:]

    if ifile > 0:

      ncidp = nc.Dataset(lalolist[ifile-1], 'r')
      ntime = len(ncidp.dimensions['Time'])
      print(nc.chartostring(ncid.variables['xtime'][0,:]))

      for varble in var_names:
        ncid[varble][0,:] = ncid[varble][0,:] - ncidp[varble][ntime-1,:]

      ncidp.close()

    ncid.close()

