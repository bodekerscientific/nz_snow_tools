import os
import xarray as xr
import datetime as dt
import numpy as np
import pickle
import matplotlib.pylab as plt

from nz_snow_tools.util.utils import setup_nztm_dem, trim_data_to_mask, trim_lat_lon_bounds


_, x_centres, y_centres, lat_array, lon_array = setup_nztm_dem(None, extent_w=1.085e6, extent_e=2.10e6, extent_n=6.20e6, extent_s=4.70e6,
                                                                          resolution=250, origin='bottomleft')
dem_folder = 'C:/Users/conwayjp/OneDrive - NIWA/Data/GIS_DATA/Topography/DEM_NZSOS'
modis_dem = 'modis_nz_dem_250m' #
nztm_dem = np.load(dem_folder + '/{}.npy'.format(modis_dem))



# alternative method
# lat_array, lon_array, nztm_dem, y_centres, x_centres = trim_lat_lon_bounds(mask, lat_array, lon_array, nztm_dem, y_centres, x_centres)

def write_fields_to_netcdf(outfile, t_scd, t_lat, t_lon, t_northing, t_easting, t_dem):
    da_scd = xr.DataArray(
        data=t_scd.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'long_name': 'annual snow cover duration',
               'units': 'days per year', })
    da_lat = xr.DataArray(
        data=t_lat.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'latitude',
               'long_name': 'latitude of grid centre',
               'units': 'degrees north', })
    da_lon = xr.DataArray(
        data=t_lon.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'longitude',
               'long_name': 'longitude of grid centre',
               'units': 'degrees east', })
    da_north = xr.DataArray(
        data=t_northing.astype('f4'),
        dims=['northing'],
        attrs={'long_name': 'NZTM northing of grid centre',
               'units': 'm', })
    da_east = xr.DataArray(
        data=t_easting.astype('f4'),
        dims=['easting'],
        attrs={'long_name': 'NZTM easting of grid centre',
               'units': 'm', })
    da_dem = xr.DataArray(
        data=t_dem.astype('f4'),
        dims=['northing', 'easting'],
        attrs={'standard_name': 'surface_altitude',
               'long_name': 'height of surface above sea level',
               'units': 'm', })
    ds = xr.Dataset({'scd': da_scd,
                     'latitude': da_lat,
                     'longitude': da_lon,
                     'easting': da_east,
                     'northing': da_north,
                     'dem': da_dem,
                     },
                    attrs={
                        'title': 'This file contains snow cover duration data derived from MODIS MOD10A supplied by Pascal Sirguey at U. Otago. Please check EULA before using niwa.local-projects-christchurch-FWWR1702-Administration-FWWR2305-MODIS data purchase-NIWA_EULA_v0 (003) Otago snow cover data signed_2019 221108.pdf',
                        'institution': 'NIWA',
                        'author': 'Jono Conway',
                        'email': 'jono.conway@niwa.co.nz',
                        'file_creation_time': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
    ds.to_netcdf(outfile)

summary_file_folder = r"\\CHRWFS02\projects$\MEL21519\Working\modis_snow_metrics"
output_folder = r"\\CHRWFS02\projects$\MEL21519\Working\modis_snow_metrics"
contents = os.listdir(summary_file_folder)
files = [s.split('.')[0] for s in contents if "summary" in s and ".pkl" in s]

for f in files:
    print('loading ' + f)
    # load scd data
    d = pickle.load(open(summary_file_folder + '/' + f + '.pkl','rb'))
    m_scd = np.mean(d[3],axis=0)
    m_scd[m_scd<0] = np.nan
    #load mask
    catchment = f.split('_')[4:11]
    # loop through each catchment:
    mask = np.load(r"\\CHRWFS02\projects$\MEL21519\Working\modis_snow_metrics\{}.npy".format('_'.join(catchment)))
    masked_lat_array = trim_data_to_mask(lat_array, mask)
    masked_lon_array = trim_data_to_mask(lon_array, mask)
    mask_y = mask.max(axis=1)
    mask_x = mask.max(axis=0)
    masked_y_centres = y_centres[mask_y]
    masked_x_centres = x_centres[mask_x]
    masked_nztm_dem = trim_data_to_mask(nztm_dem, mask)

    write_fields_to_netcdf(output_folder + '/' +'scd_' + f + '.nc', m_scd, masked_lat_array, masked_lon_array, masked_y_centres, masked_x_centres, masked_nztm_dem)

    d = xr.load_dataset(output_folder + '/' +'scd_' + f + '.nc')
    d.scd.plot(vmin=0,vmax=365)
    plt.gca().set_aspect('equal')
    plt.title(f,fontsize=8)
    plt.tight_layout()
    plt.savefig(output_folder + '/' +'scd_' + f + '.png',dpi=300)
    plt.close()
