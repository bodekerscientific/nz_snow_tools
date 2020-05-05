

import numpy as np
import pickle
import matplotlib.pylab as plt
import datetime as dt
from nz_snow_tools.util.utils import convert_datetime_julian_day


catchment = 'Tar_Patea_NZTM'  # string identifying catchment modelled
modis_dem = 'modis_nz_dem_250m'  # identifier for output dem
years_to_take = np.arange(2010, 2010 + 1)  # range(2016, 2016 + 1)  # [2013 + 1]  # range(2001, 2013 + 1)
modis_sc_threshold = 50  # value of fsca (in percent) that is counted as being snow covered
output_folder = '/nesi/nobackup/niwa00026/Observation/Snow_RemoteSensing/catchment_output'

[ann_ts_av_sca_m, ann_ts_av_sca_thres_m, ann_dt_m, ann_scd_m] = pickle.load(open(
        output_folder + '/summary_MODIS_{}_{}_{}_{}_thres{}.pkl'.format(years_to_take[0], years_to_take[-1], catchment, modis_dem,
                                                                        modis_sc_threshold),'rb'))
for ts_av_sca_m, dt_m in zip(ann_ts_av_sca_m, ann_dt_m):
    plt.plot(convert_datetime_julian_day(dt_m), ts_av_sca_m,label=dt_m[0].year)

ax = plt.gca()
ax.set_ylim([0, 1])
ax.set_ylabel('SCA')
ax.set_xlabel('day of year')
plt.show()