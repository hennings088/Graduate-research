############
# creates an overview plot showing the state of the atmosphere of a section of the northern hemisphere mid-latitudes at a specific point in time.
# includes ivt, vorticity, and the location of the PV waveguide
# figure is located in analfig folder
############


import sys, os
import glob
import numpy as np
import xarray as xr
import cfgrib
import argparse
import urllib.request
import datetime as dt
from herbie import Herbie
import matplotlib
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import metpy.constants as mpcon
import metpy.calc as mpcalc
from metpy.units import units

sys.path.append('/home11/staff/torn/ens-sens/esens-util')
import grid_calc

import types
from windspharm.xarray import VectorWind, _find_latitude_coordinate, _find_longitude_coordinate, _reverse
from windspharm._common import get_apiorder, inspect_gridtype, to3d


def getuv(self, vorticity, divergence):
    """Compute vector winds from vorticity and divergence fields.
        Method to attach to windspharm.standard.vectorWind
        This is kinda hacky but windspharm development seems dead.
    **Argument:**
    *vorticity*
        A scalar field of vorticity. Its shape must be either (nlat, nlon) or
        (nlat, nlon, nfields) where nlat and nlon are the same
        as those for the vector wind components that initialized the
        `VectorWind` instance.
    *divergence*
        A scalar field of divergence. Its shape must be either (nlat, nlon) or
        (nlat, nlon, nfields) where nlat and nlon are the same
        as those for the vector wind components that initialized the
        `VectorWind` instance.
    **Optional argument:**
    **Returns:**
    *u*, *v*
        Zonal and meridional wind components respectively. Their types will 
        match input types to passed to `VectorWind` instance. 
    """
    vrspec = self.s.grdtospec(vorticity)
    dvspec = self.s.grdtospec(divergence)
    ugrd, vgrid = self.s.getuv(vrspec, dvspec)
    return ugrd, vgrid


class ExtendedVectorWind(VectorWind):
    def __init__(self, u, v, rsphere=6.3712e6, legfunc='stored'):
        super().__init__(u, v, rsphere, legfunc)
        self._api.getuv = types.MethodType(getuv, self._api)

    def getuv(self, vorticity, divergence):
         """Compute vector winds from vorticity and divergence fields.
         **Argument:**
         *vorticity*
             A scalar field of vorticity. Its shape must be either (nlat, nlon) or
             (nlat, nlon, nfields) where nlat and nlon are the same
             as those for the vector wind components that initialized the
             `VectorWind` instance.
         *divergence*
             A scalar field of divergence. Its shape must be either (nlat, nlon) or
             (nlat, nlon, nfields) where nlat and nlon are the same
             as those for the vector wind components that initialized the
             `VectorWind` instance.
         **Returns:**
         *u*, *v*
             Zonal and meridional wind components respectively. Their types will 
             match input types to passed to `VectorWind` instance. 
         """
         def clean_array(field):
             if not isinstance(field, xr.DataArray):
                 raise TypeError('scalar field must be an xarray.DataArray')
             name = field.name
             lat, lat_dim = _find_latitude_coordinate(field)
             lon, lon_dim = _find_longitude_coordinate(field)
             if (lat.values[0] < lat.values[1]):
                 # need to reverse latitude dimension
                 field = _reverse(field, lat_dim)
                 lat, lat_dim = _find_latitude_coordinate(field)
             apiorder, _ = get_apiorder(field.ndim, lat_dim, lon_dim)
             apiorder = [field.dims[i] for i in apiorder]
             reorder = field.dims
             field = field.copy().transpose(*apiorder)
             ishape = field.shape
             coords = [field.coords[n] for n in field.dims]
             field = to3d(field.values)
             return field

         vortic = clean_array(vorticity)
         diverg = clean_array(divergence)
         vortic[:,:] = 0.0

         ugrd, vgrd = self._api.getuv(vortic, diverg)
         ugrd = self._metadata(ugrd, 'u',
                     units='m s**-1',
                     standard_name='eastward_wind',
                     long_name='eastward_component_of_wind')
         vgrd = self._metadata(vgrd, 'v',
                     units='m s**-1',
                     standard_name='northward_wind',
                     long_name='northward_component_of_wind')
         return ugrd, vgrd



def background_map(ax):

  states = NaturalEarthFeature(category="cultural", scale="50m",
                               facecolor="none", name="admin_1_states_provinces")
  ax.add_feature(states, linewidth=0.5, edgecolor="black")
  ax.coastlines('50m', linewidth=1.0)
  ax.add_feature(cartopy.feature.LAKES, linewidth=1.0, edgecolor='black', facecolor='none')
  ax.add_feature(cartopy.feature.BORDERS, linewidth=1.0, edgecolor='black')

  gridInt = 5.0

  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='-')
  gl.top_labels = None
  gl.bottom_labels = True
  gl.left_labels = True
  gl.right_labels = None
  gl.xlocator = mticker.FixedLocator(np.arange(-180.,180.,gridInt))
  gl.xformatter = LONGITUDE_FORMATTER
  gl.xlabel_style = {'size': 12, 'color': 'gray'}
  gl.ylocator = mticker.FixedLocator(np.arange(-90.+gridInt,90.,gridInt))
  gl.yformatter = LATITUDE_FORMATTER
  gl.ylabel_style = {'size': 12, 'color': 'gray'}

  return(ax)


def read_grib_file(file_name):

  try:
    ds = cfgrib.open_datasets(file_name)
    grib_dict = {}
    for d in ds:
      for tt in d:
        grib_dict.update({'{0}'.format(tt): d[tt]})

  except IOError as exc:
    raise RuntimeError('Failed to open {0}'.format(file_name)) from exc

  return(grib_dict)


def calc_ivt(fdict, lam1, lom1, lam2, lom2):

  #  Read wind/temperature data 
  uinp = fdict['u'].sel(latitude=slice(lam2, lam1), longitude=slice(lom1, lom2), isobaricInhPa=slice(1000, 300)) * units('m/s')
  vinp = fdict['v'].sel(latitude=slice(lam2, lam1), longitude=slice(lom1, lom2), isobaricInhPa=slice(1000, 300)) * units('m/s')

  if 'q' in fdict:
    qinp = fdict['q'].sel(latitude=slice(lam2, lam1), longitude=slice(lom1, lom2), isobaricInhPa=slice(1000, 300)).squeeze()
    pinp = (qinp.isobaricInhPa.values * units.hPa).to(units.Pa)
  else:
    tinp = fdict['t'].sel(latitude=slice(lam2, lam1), longitude=slice(lom1, lom2), isobaricInhPa=slice(1000, 300)).squeeze() * units('K')
    pinp = (tinp.isobaricInhPa.values * units.hPa).to(units.Pa)
    rinp = np.minimum(np.maximum(fdict['r'].sel(latitude=slice(lam2, lam1), longitude=slice(lom1, lom2), isobaricInhPa=slice(1000, 300)).squeeze(), 0.01), 100.0) * units('percent')
    qinp = mpcalc.mixing_ratio_from_relative_humidity(pinp[:,None,None], tinp, rinp)

    del tinp, rinp

  #  Integrate water vapor over the pressure levels
  ivtu = np.abs(np.trapezoid(uinp[:,:,:].data*qinp[:,:,:].data, pinp, axis=0)) / mpcon.earth_gravity
  ivtv = np.abs(np.trapezoid(vinp[:,:,:].data*qinp[:,:,:].data, pinp, axis=0)) / mpcon.earth_gravity
  ivtt = np.sqrt(ivtu[:,:]**2 + ivtv[:,:]**2)

  del uinp, vinp, pinp, qinp

  return ivtt


def calc_vint_wind(fdict, lat1, lat2, lon1, lon2):

  uwnd = fdict['u'].sel(isobaricInhPa=slice(300,200)).squeeze()
  vwnd = fdict['v'].sel(isobaricInhPa=slice(300,200)).squeeze()

  pres = uwnd.isobaricInhPa.values

  uint      = uwnd[0,:,:].copy().squeeze()
  uint[:,:] = 0.0
  vint      = vwnd[0,:,:].copy().squeeze()
  vint[:,:] = 0.0

  for k in range(len(pres)-1):
    uint[:,:] = uint[:,:] + 0.5 * (uwnd[k,:,:]+uwnd[k+1,:,:]) * abs(pres[k+1]-pres[k])
    vint[:,:] = vint[:,:] + 0.5 * (vwnd[k,:,:]+vwnd[k+1,:,:]) * abs(pres[k+1]-pres[k])

  uint[:,:] = uint[:,:] / abs(pres[-1]-pres[0])
  vint[:,:] = vint[:,:] / abs(pres[-1]-pres[0])

  return(uint,vint)

def calc_vert_int(infld):

  pres = infld.isobaricInhPa.values

  vintfld = infld[0,:,:].copy().squeeze().values
  vintfld[:,:] = 0.0

  for k in range(len(pres)-1):
    vintfld[:,:] = vintfld[:,:] + 0.5 * (infld[k,:,:].values+infld[k+1,:,:].values) * abs(pres[k+1]-pres[k])

  vintfld[:,:] = vintfld[:,:] / abs(pres[-1]-pres[0])

  return (vintfld)


def model_anal(init, hour):

  basedir = '.'

  lat1 = 30 
  lat2 = 75
  lon1 = 150
  lon2 = 330
  intype = 'gfs'

  #  Read the relevant forecast field here
  barbInt = 6

  initstr = '{0}-{1}-{2} {3}:00'.format(init[0:4],init[4:6],init[6:8],init[8:10])
  initdt  = dt.datetime.strptime(init, '%Y%m%d%H')
  fcstdt  = initdt + dt.timedelta(hours=hour)
  fcst_str = fcstdt.strftime("%Y-%m-%d_%H.%M.00")

  if intype == 'ecmwf':

     initdt  = dt.datetime.strptime(init, '%Y%m%d%H')
     datef   = initdt + dt.timedelta(hours=hour)
     grib_file = '/tornlab_rit/torn/ecmwf/E1E{0}{1}1'.format(str(initdt.strftime("%m%d%H%M")),str(datef.strftime("%m%d%H%M")))
#     grib_file = '/rt11/torn/ar_sens/east_ecmwf/2017010612/E1E{0}{1}1'.format(str(initdt.strftime("%m%d%H%M")),str(datef.strftime("%m%d%H%M")))
     slpname = 'msl'

  elif intype == 'gfs':

     initdt  = dt.datetime.strptime(init, '%Y%m%d%H')
     fhrt    = '%0.3i' % hour

     varstr1 = ':UGRD:\\d+ mb|:VGRD:\\d+ mb|:HGT:\\d+ mb|:RH:\\d+ mb|:TMP:\\d+ mb|:UGRD:10 m above|:VGRD:10 m above|PRMSL|PWAT|APCP|CFRZR|CSNOW'
     varstr2 = 'TMP:300 mb|TMP:400 mb|RH:300 mb|RH:400 mb'

     #  Read fields from primary file
     if initdt < dt.datetime.strptime('2021010100', '%Y%m%d%H'):
        urllib.request.urlretrieve('https://data-osdf.rda.ucar.edu/ncar/rda/d084001/{0}/{1}/gfs.0p25.{2}.f{3}.grib2'.format(init[0:4],init[0:8],init,fhrt), \
                                   'gfs.grb')
        grib_file = 'gfs.grb'       

     else:

        H = Herbie(initstr, model='gfs', product='pgrb2.0p50', fxx=hour)
        file1 = H.download(varstr1)

        #  Merge output from primary and secondary file, remove single member files
        os.system('cat {0} >> {1}_{2}_gfs.grb2'.format(file1,init,fhrt))
        os.remove(file1)
        grib_file = '{0}_{1}_gfs.grb2'.format(init, fhrt)
     
     slpname = 'prmsl'


  fdict = read_grib_file(grib_file)

  fig = plt.figure(figsize=(10, 10))

  ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.))
  #ax = plt.axes(projection=ccrs.PlateCarree())
  ax = background_map(ax)
  ax.set_extent([lon1, lon2, lat1, lat2], ccrs.PlateCarree())

  ivt = calc_ivt(fdict, lat1, lon1, lat2, lon2)
  latmod = fdict['u'].sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2)).latitude.values
  lonmod = fdict['u'].sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2)).longitude.values

  colorlist = ("#FFFFFF", "#FFFC00", "#FFE100", "#FFC600", "#FFAA00", "#FF7D00", \
               "#FF4B00", "#FF1900", "#E60015", "#B3003E", "#80007B", "#570088")
  mivt = [0.0, 250., 300., 400., 500., 600., 700., 800., 1000., 1200., 1400., 1600., 2000.]
  norm = matplotlib.colors.BoundaryNorm(mivt,len(mivt))
  plti = plt.contourf(lonmod,latmod,ivt,mivt,transform=ccrs.PlateCarree(), \
                       cmap=matplotlib.colors.ListedColormap(colorlist), norm=norm, extend='max')

  uwnd,vwnd = calc_vint_wind(fdict, lat1, lat2, lon1, lon2)

  ws_wnd = ExtendedVectorWind(uwnd, vwnd)

  vr = ws_wnd.vorticity()
  dv = ws_wnd.divergence()
  vr[:,:] = 0.0

  udiv, vdiv = ws_wnd.getuv(vr, dv)
  mdiv = np.sqrt(udiv[:,:]**2 + vdiv[:,:]**2)
  udiv[:,:] = np.where(mdiv >= 5.0, udiv, np.nan)
  vdiv[:,:] = np.where(mdiv >= 5.0, vdiv, np.nan)

#    pltm = plt.barbs(uwnd.longitude.values[::barbInt], uwnd.latitude.values[::barbInt], \
#                     udiv[::barbInt,::barbInt].values*1.94, vdiv[::barbInt,::barbInt].values*1.94, \
#                     color='gray', pivot='middle', length=6, linewidths=0.2, zorder=0, transform=ccrs.PlateCarree())

  sout = 150.
  qo1 = ax.quiver(udiv.longitude.values[::barbInt], udiv.latitude.values[::barbInt], \
                  udiv[::barbInt,::barbInt].values, vdiv[::barbInt,::barbInt].values, \
                  scale_units='height', scale=sout, width=0.005, pivot='mid', color='gray', minlength=0, transform=ccrs.PlateCarree())
  l, b, w, h = ax.get_position().bounds
  qk = ax.quiverkey(qo1, l+w-0.05, b+0.07, 10, '10', labelpos='E', coordinates='figure')

  dv = dv.sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2))
  lats = dv.latitude.values * units('degrees')
  lons = dv.longitude.values * units('degrees')
  dvplt = grid_calc.calc_circ_llgrid(dv, 100., lats, lons, False, len(lons), len(lats)) * 1.0e5

#    pltm = plt.contour(lons, lats, dvplt, [-8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0], \
#                        linewidths=1.5, colors='k', zorder=5, transform=ccrs.PlateCarree())

#    cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt='%3.1f')

  mslp = fdict[slpname].sel(latitude=slice(lat2, lat1), longitude=slice(lon1, lon2)).squeeze()
  print("mslp shape:", mslp.shape)
  print("mslp dims:", mslp.dims)
  pltm = plt.contour(mslp.longitude.values, mslp.latitude.values, mslp*0.01, np.arange(960,1044,4), \
                     linewidths=1.0, colors='k', zorder=5, transform=ccrs.PlateCarree())

  cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12)


  pvec = fdict['t'].isobaricInhPa.values
#  idx  = np.where(pvec==250)
#  lev1 = np.min(pvec[(int(idx[0])-1):(int(idx[0])+2)])
#  lev2 = np.max(pvec[(int(idx[0])-1):(int(idx[0])+2)])
  idx  = np.where(pvec==200)
  lev1 = np.min(pvec[(int(idx[0])-1):(int(idx[0])+2)])
  idx  = np.where(pvec==300)
  lev2 = np.max(pvec[(int(idx[0])-1):(int(idx[0])+2)])

  #  Read all the necessary files from file, smooth fields, so sensitivities are useful
  tmpk = fdict['t'].sel(latitude=slice(lat2,lat1),longitude=slice(lon1,lon2),isobaricInhPa=slice(lev2,lev1)).reset_coords('valid_time',drop=True) * units('K')

  lats = tmpk.latitude.values * units('degrees')
  lons = tmpk.longitude.values * units('degrees')
  pres = tmpk.isobaricInhPa.values * units('hPa')

  thta = mpcalc.potential_temperature(pres[:, None, None], tmpk)

  uwnd = fdict['u'].sel(latitude=slice(lat2,lat1),longitude=slice(lon1,lon2),isobaricInhPa=slice(lev2,lev1)).reset_coords('valid_time',drop=True) * units('m/s')
  vwnd = fdict['v'].sel(latitude=slice(lat2,lat1),longitude=slice(lon1,lon2),isobaricInhPa=slice(lev2,lev1)).reset_coords('valid_time',drop=True) * units('m/s')

  dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats, x_dim=-1, y_dim=-2, geod=None)

  pvout = mpcalc.potential_vorticity_baroclinic(thta, pres[:, None, None], uwnd, vwnd,
                                              dx[None, :, :], dy[None, :, :], lats[None, :, None])

  pvint = calc_vert_int(pvout[1:-1,:,:])
  pvplt = grid_calc.calc_circ_llgrid(pvint, 100., lats, lons, False, len(lons), len(lats)) * 1.0e6

#  pltm = plt.contour(tmpk.longitude.values, tmpk.latitude.values, pvplt, np.arange(2.0, 8.0, 2.0),\
#                      linewidths=2.0, colors='limegreen', zorder=5, transform=ccrs.PlateCarree())
  pltm = plt.contour(tmpk.longitude.values, tmpk.latitude.values, pvplt, [2.0],\
                      linewidths=4.0, colors='limegreen', zorder=5, transform=ccrs.PlateCarree())

  plth = plt.contourf(tmpk.longitude.values, tmpk.latitude.values, np.ma.masked_array(pvplt, mask=(pvplt<5.)), [5.0, 7.5], alpha=0.3, \
                      cmap=matplotlib.colors.ListedColormap(["#FFFFFF", "#0096FF", "#0818A8"]), extend='both', \
                      zorder=1, antialiased=True, transform=ccrs.PlateCarree())
#  plt1 = plt.contour(tmpk.longitude.values, tmpk.latitude.values, pvplt, [5.0], linewidths=0.5, \
#                        colors='blue', zorder=5, transform=ccrs.PlateCarree())
#  plth = plt.contourf(tmpk.longitude.values, tmpk.latitude.values, pvplt, [-5.0, 5.0], hatches=['/', None, '/'], \
#                      colors='none', extend='both', zorder=5, transform=ccrs.PlateCarree())
#  for i, collection in enumerate(plth.collections):
#     collection.set_edgecolor('blue')
#  for collection in plth.collections:
#     collection.set_linewidth(0.)

#  cb = plt.clabel(pltm, inline_spacing=0.0, fontsize=12, fmt='%3.1f')

#  plt.title('{0} {1} hPa {2}'.format(titletime, int(args.pressure), titlename))

  plt.title(fcst_str)

  plt.savefig('analfig/{0}_{1}_grid.png'.format(init,'%0.3i' % hour),format='png',dpi=150,bbox_inches='tight')
  plt.close(fig)

  if intype == 'gfs':
    for f in glob.glob('{0}_{1}_gfs.grb2*'.format(init,fhrt)):
      os.remove(f)


if __name__ == '__main__':

  #  Read the initialization time and storm from the command line
  exp_parser = argparse.ArgumentParser()
  exp_parser.add_argument('--init', action='store', type=str, required=True)
  exp_parser.add_argument('--hour', action='store', type=int, required=True)
  args  = exp_parser.parse_args()

  model_anal(args.init, args.hour)
