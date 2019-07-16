"""
code to plot with the subplots the different models for each year and each station
"""

import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
import netCDF4 as nc

#MUELLER [2011-2018]

# csv file
csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv"
# npy files
# clark_file with different VCSN parameters
clark_file_obs = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/Observed precipitation/{}_clark2009_{}.npy"
clark_file_precip_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + ta/{}_clark2009_{}.npy"
clark_file_precip_rad = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + rad/{}_clark2009_{}.npy"
clark_file_precip_rad_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + rad + ta/{}_clark2009_{}.npy"
clark_file_precip = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip/{}_clark2009_{}.npy"
clark_file_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN ta/{}_clark2009_{}.npy"
clark_file_rad = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN rad/{}_clark2009_{}.npy"
clark_file_rad_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN rad + ta/{}_clark2009_{}.npy"
# albedo_file with different VCSN parameters
albedo_file_obs = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/Observed precipitation/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_rad = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + rad/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip_rad_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip + rad + ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_precip = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN precip/{}_dsc_snow-param albedo_{}.npy"
albedo_file_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN ta/{}_dsc_snow-param albedo_{}.npy"
albedo_file_rad = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN rad/{}_dsc_snow-param albedo_{}.npy"
albedo_file_rad_ta = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/{}/{}_npy files/VCSN rad + ta/{}_dsc_snow-param albedo_{}.npy"

Stname = ['Mueller']
year0 = 2011
year1 = 2012
year2 = 2013
year3 = 2014
year4 = 2015
year5 = 2016
year6 = 2017
year7 = 2018

# load npy data
# observed data
# clark
inp_clark_obs0 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_clark_obs1 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_clark_obs2 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_clark_obs3 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_clark_obs4 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_clark_obs5 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_clark_obs6 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_clark_obs7 = np.load(clark_file_obs.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
# inp_dt_obs = inp_clark_obs[:, 0]  # model stores initial state
# inp_swe_obs = np.asarray(inp_clark_obs[:, 1], dtype=np.float)
plot_dt0 = inp_clark_obs0[:, 0] # model stores initial state
plot_dt1 = inp_clark_obs1[:, 0] # model stores initial state
plot_dt2 = inp_clark_obs2[:, 0] # model stores initial state
plot_dt3 = inp_clark_obs3[:, 0] # model stores initial state
plot_dt4 = inp_clark_obs4[:, 0] # model stores initial state
plot_dt5 = inp_clark_obs5[:, 0] # model stores initial state
plot_dt6 = inp_clark_obs6[:, 0] # model stores initial state
plot_dt7 = inp_clark_obs7[:, 0] # model stores initial state
# albedo
inp_alb_obs0 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_alb_obs1 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_alb_obs2 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_alb_obs3 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_alb_obs4 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_alb_obs5 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_alb_obs6 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_alb_obs7 = np.load(albedo_file_obs.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
inp_dt1_obs0 = inp_alb_obs0[:, 0]  # model stores initial state
inp_dt1_obs1 = inp_alb_obs1[:, 0]  # model stores initial state
inp_dt1_obs2 = inp_alb_obs2[:, 0]  # model stores initial state
inp_dt1_obs3 = inp_alb_obs3[:, 0]  # model stores initial state
inp_dt1_obs4 = inp_alb_obs4[:, 0]  # model stores initial state
inp_dt1_obs5 = inp_alb_obs5[:, 0]  # model stores initial state
inp_dt1_obs6 = inp_alb_obs6[:, 0]  # model stores initial state
inp_dt1_obs7 = inp_alb_obs7[:, 0]  # model stores initial state
inp_swe1_obs0 = np.asarray(inp_alb_obs0[:, 1], dtype=np.float)
inp_swe1_obs1 = np.asarray(inp_alb_obs1[:, 1], dtype=np.float)
inp_swe1_obs2 = np.asarray(inp_alb_obs2[:, 1], dtype=np.float)
inp_swe1_obs3 = np.asarray(inp_alb_obs3[:, 1], dtype=np.float)
inp_swe1_obs4 = np.asarray(inp_alb_obs4[:, 1], dtype=np.float)
inp_swe1_obs5 = np.asarray(inp_alb_obs5[:, 1], dtype=np.float)
inp_swe1_obs6 = np.asarray(inp_alb_obs6[:, 1], dtype=np.float)
inp_swe1_obs7 = np.asarray(inp_alb_obs7[:, 1], dtype=np.float)

# VCSN precipitation + temperature
# clark
# inp_clark_precip_ta = np.load(clark_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_precip_ta = inp_clark_precip_ta[:, 0]  # model stores initial state
# inp_swe_VC_precip_ta = np.asarray(inp_clark_precip_ta[:, 1], dtype=np.float)
# inp_swe_VN_precip_ta = np.asarray(inp_clark_precip_ta[:, 2], dtype=np.float)
# albedo
inp_alb_precip_ta0 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_alb_precip_ta1 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_alb_precip_ta2 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_alb_precip_ta3 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_alb_precip_ta4 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_alb_precip_ta5 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_alb_precip_ta6 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_alb_precip_ta7 = np.load(albedo_file_precip_ta.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
inp_dt1_precip_ta0 = inp_alb_precip_ta0[:, 0]  # model stores initial state
inp_dt1_precip_ta1 = inp_alb_precip_ta1[:, 0]  # model stores initial state
inp_dt1_precip_ta2 = inp_alb_precip_ta2[:, 0]  # model stores initial state
inp_dt1_precip_ta3 = inp_alb_precip_ta3[:, 0]  # model stores initial state
inp_dt1_precip_ta4 = inp_alb_precip_ta4[:, 0]  # model stores initial state
inp_dt1_precip_ta5 = inp_alb_precip_ta5[:, 0]  # model stores initial state
inp_dt1_precip_ta6 = inp_alb_precip_ta6[:, 0]  # model stores initial state
inp_dt1_precip_ta7 = inp_alb_precip_ta7[:, 0]  # model stores initial state
inp_swe1_VC_precip_ta0 = np.asarray(inp_alb_precip_ta0[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta1 = np.asarray(inp_alb_precip_ta1[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta2 = np.asarray(inp_alb_precip_ta2[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta3 = np.asarray(inp_alb_precip_ta3[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta4 = np.asarray(inp_alb_precip_ta4[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta5 = np.asarray(inp_alb_precip_ta5[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta6 = np.asarray(inp_alb_precip_ta6[:, 1], dtype=np.float)
inp_swe1_VC_precip_ta7 = np.asarray(inp_alb_precip_ta7[:, 1], dtype=np.float)
inp_swe1_VN_precip_ta0 = np.asarray(inp_alb_precip_ta0[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta1 = np.asarray(inp_alb_precip_ta1[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta2 = np.asarray(inp_alb_precip_ta2[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta3 = np.asarray(inp_alb_precip_ta3[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta4 = np.asarray(inp_alb_precip_ta4[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta5 = np.asarray(inp_alb_precip_ta5[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta6 = np.asarray(inp_alb_precip_ta6[:, 2], dtype=np.float)
inp_swe1_VN_precip_ta7 = np.asarray(inp_alb_precip_ta7[:, 2], dtype=np.float)

# VCSN precipitation + radiation
# clark
# inp_clark_precip_rad = np.load(clark_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_precip_rad = inp_clark_precip_rad[:, 0]  # model stores initial state
# inp_swe_VC_precip_rad = np.asarray(inp_clark_precip_rad[:, 1], dtype=np.float)
# inp_swe_VN_precip_rad = np.asarray(inp_clark_precip_rad[:, 2], dtype=np.float)
# albedo
inp_alb_precip_rad0 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_alb_precip_rad1 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_alb_precip_rad2 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_alb_precip_rad3 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_alb_precip_rad4 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_alb_precip_rad5 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_alb_precip_rad6 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_alb_precip_rad7 = np.load(albedo_file_precip_rad.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
inp_dt1_precip_rad0 = inp_alb_precip_rad0[:, 0]  # model stores initial state
inp_dt1_precip_rad1 = inp_alb_precip_rad1[:, 0]  # model stores initial state
inp_dt1_precip_rad2 = inp_alb_precip_rad2[:, 0]  # model stores initial state
inp_dt1_precip_rad3 = inp_alb_precip_rad3[:, 0]  # model stores initial state
inp_dt1_precip_rad4 = inp_alb_precip_rad4[:, 0]  # model stores initial state
inp_dt1_precip_rad5 = inp_alb_precip_rad5[:, 0]  # model stores initial state
inp_dt1_precip_rad6 = inp_alb_precip_rad6[:, 0]  # model stores initial state
inp_dt1_precip_rad7 = inp_alb_precip_rad7[:, 0]  # model stores initial state
inp_swe1_VC_precip_rad0 = np.asarray(inp_alb_precip_rad0[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad1 = np.asarray(inp_alb_precip_rad1[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad2 = np.asarray(inp_alb_precip_rad2[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad3 = np.asarray(inp_alb_precip_rad3[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad4 = np.asarray(inp_alb_precip_rad4[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad5 = np.asarray(inp_alb_precip_rad5[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad6 = np.asarray(inp_alb_precip_rad6[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad7 = np.asarray(inp_alb_precip_rad7[:, 1], dtype=np.float)
inp_swe1_VN_precip_rad0 = np.asarray(inp_alb_precip_rad0[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad1 = np.asarray(inp_alb_precip_rad1[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad2 = np.asarray(inp_alb_precip_rad2[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad3 = np.asarray(inp_alb_precip_rad3[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad4 = np.asarray(inp_alb_precip_rad4[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad5 = np.asarray(inp_alb_precip_rad5[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad6 = np.asarray(inp_alb_precip_rad6[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad7 = np.asarray(inp_alb_precip_rad7[:, 2], dtype=np.float)

# VCSN precipitation + radiation + temperature
# clark
# inp_clark_precip_rad_ta = np.load(clark_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_precip_rad_ta = inp_clark_precip_rad_ta[:, 0]  # model stores initial state
# inp_swe_VC_precip_rad_ta = np.asarray(inp_clark_precip_rad_ta[:, 1], dtype=np.float)
# inp_swe_VN_precip_rad_ta = np.asarray(inp_clark_precip_rad_ta[:, 2], dtype=np.float)
# albedo
inp_alb_precip_rad_ta0 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_alb_precip_rad_ta1 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_alb_precip_rad_ta2 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_alb_precip_rad_ta3 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_alb_precip_rad_ta4 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_alb_precip_rad_ta5 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_alb_precip_rad_ta6 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_alb_precip_rad_ta7 = np.load(albedo_file_precip_rad_ta.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
inp_dt1_precip_rad_ta0 = inp_alb_precip_rad_ta0[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta1 = inp_alb_precip_rad_ta1[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta2 = inp_alb_precip_rad_ta2[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta3 = inp_alb_precip_rad_ta3[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta4 = inp_alb_precip_rad_ta4[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta5 = inp_alb_precip_rad_ta5[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta6 = inp_alb_precip_rad_ta6[:, 0]  # model stores initial state
inp_dt1_precip_rad_ta7 = inp_alb_precip_rad_ta7[:, 0]  # model stores initial state
inp_swe1_VC_precip_rad_ta0 = np.asarray(inp_alb_precip_rad_ta0[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta1 = np.asarray(inp_alb_precip_rad_ta1[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta2 = np.asarray(inp_alb_precip_rad_ta2[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta3 = np.asarray(inp_alb_precip_rad_ta3[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta4 = np.asarray(inp_alb_precip_rad_ta4[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta5 = np.asarray(inp_alb_precip_rad_ta5[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta6 = np.asarray(inp_alb_precip_rad_ta6[:, 1], dtype=np.float)
inp_swe1_VC_precip_rad_ta7 = np.asarray(inp_alb_precip_rad_ta7[:, 1], dtype=np.float)
inp_swe1_VN_precip_rad_ta0 = np.asarray(inp_alb_precip_rad_ta0[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta1 = np.asarray(inp_alb_precip_rad_ta1[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta2 = np.asarray(inp_alb_precip_rad_ta2[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta3 = np.asarray(inp_alb_precip_rad_ta3[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta4 = np.asarray(inp_alb_precip_rad_ta4[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta5 = np.asarray(inp_alb_precip_rad_ta5[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta6 = np.asarray(inp_alb_precip_rad_ta6[:, 2], dtype=np.float)
inp_swe1_VN_precip_rad_ta7 = np.asarray(inp_alb_precip_rad_ta7[:, 2], dtype=np.float)

# VCSN precipitation
# clark
# inp_clark_precip = np.load(clark_file_precip.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_precip = inp_clark_precip[:, 0]  # model stores initial state
# inp_swe_precip = np.asarray(inp_clark_precip[:, 1], dtype=np.float)
# albedo
inp_alb_precip0 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year0), allow_pickle=True)
inp_alb_precip1 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year1), allow_pickle=True)
inp_alb_precip2 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year2), allow_pickle=True)
inp_alb_precip3 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year3), allow_pickle=True)
inp_alb_precip4 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year4), allow_pickle=True)
inp_alb_precip5 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year5), allow_pickle=True)
inp_alb_precip6 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year6), allow_pickle=True)
inp_alb_precip7 = np.load(albedo_file_precip.format(Stname[0],Stname[0],Stname[0],year7), allow_pickle=True)
inp_dt1_precip0 = inp_alb_precip0[:, 0]  # model stores initial state
inp_dt1_precip1 = inp_alb_precip1[:, 0]  # model stores initial state
inp_dt1_precip2 = inp_alb_precip2[:, 0]  # model stores initial state
inp_dt1_precip3 = inp_alb_precip3[:, 0]  # model stores initial state
inp_dt1_precip4 = inp_alb_precip4[:, 0]  # model stores initial state
inp_dt1_precip5 = inp_alb_precip5[:, 0]  # model stores initial state
inp_dt1_precip6 = inp_alb_precip6[:, 0]  # model stores initial state
inp_dt1_precip7 = inp_alb_precip7[:, 0]  # model stores initial state
inp_swe1_precip0 = np.asarray(inp_alb_precip0[:, 1], dtype=np.float)
inp_swe1_precip1 = np.asarray(inp_alb_precip1[:, 1], dtype=np.float)
inp_swe1_precip2 = np.asarray(inp_alb_precip2[:, 1], dtype=np.float)
inp_swe1_precip3 = np.asarray(inp_alb_precip3[:, 1], dtype=np.float)
inp_swe1_precip4 = np.asarray(inp_alb_precip4[:, 1], dtype=np.float)
inp_swe1_precip5 = np.asarray(inp_alb_precip5[:, 1], dtype=np.float)
inp_swe1_precip6 = np.asarray(inp_alb_precip6[:, 1], dtype=np.float)
inp_swe1_precip7 = np.asarray(inp_alb_precip7[:, 1], dtype=np.float)

# VCSN temperature
# clark
# inp_clark_ta = np.load(clark_file_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_ta = inp_clark_ta[:, 0]  # model stores initial state
# inp_swe_VC_ta = np.asarray(inp_clark_ta[:, 1], dtype=np.float)
# inp_swe_VN_ta = np.asarray(inp_clark_ta[:, 2], dtype=np.float)
# albedo
inp_alb_ta = np.load(albedo_file_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
inp_dt1_ta = inp_alb_ta[:, 0]  # model stores initial state
inp_swe1_VC_ta = np.asarray(inp_alb_ta[:, 1], dtype=np.float)
inp_swe1_VN_ta = np.asarray(inp_alb_ta[:, 2], dtype=np.float)

# VCSN radiation
# clark
# inp_clark_rad = np.load(clark_file_rad.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_rad = inp_clark_rad[:, 0]  # model stores initial state
# inp_swe_VC_rad = np.asarray(inp_clark_rad[:, 1], dtype=np.float)
# inp_swe_VN_rad = np.asarray(inp_clark_rad[:, 2], dtype=np.float)
# albedo
inp_alb_rad = np.load(albedo_file_rad.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
inp_dt1_rad = inp_alb_rad[:, 0]  # model stores initial state
inp_swe1_VC_rad = np.asarray(inp_alb_rad[:, 1], dtype=np.float)
inp_swe1_VN_rad = np.asarray(inp_alb_rad[:, 2], dtype=np.float)

# VCSN radiation + temperature
# clark
# inp_clark_rad_ta = np.load(clark_file_rad_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
# inp_dt_rad_ta = inp_clark_rad_ta[:, 0]  # model stores initial state
# inp_swe_VC_rad_ta = np.asarray(inp_clark_rad_ta[:, 1], dtype=np.float)
# inp_swe_VN_rad_ta = np.asarray(inp_clark_rad_ta[:, 2], dtype=np.float)
# albedo
inp_alb_rad_ta = np.load(albedo_file_rad_ta.format(Stname[0],Stname[0],Stname[0],year), allow_pickle=True)
inp_dt1_rad_ta = inp_alb_rad_ta[:, 0]  # model stores initial state
inp_swe1_VC_rad_ta = np.asarray(inp_alb_rad_ta[:, 1], dtype=np.float)
inp_swe1_VN_rad_ta = np.asarray(inp_alb_rad_ta[:, 2], dtype=np.float)


# load csv file
inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),skip_header=4)*1000
inp_timeobs = np.genfromtxt(csv_file, usecols=(0),dtype=(str), delimiter=',', skip_header=4)
inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])
ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])
datobs_year = inp_datobs[ind]
dtobs_year = inp_dtobs[ind]

# plot
ax0 = plt.subplot(811)
plt.plot(dtobs_year, datobs_year, 'o', label = 'Observed')
plt.plot(inp_dt1_obs0, inp_swe1_obs0,'*', label = 'Obs precip')
plt.plot(inp_dt1_precip, inp_swe1_precip, '.', color = 'firebrick', label = 'VCSN precip')

plt.plot(inp_dt1_precip_ta, inp_swe1_VC_precip_ta, linewidth = 1, color = 'seagreen', label = 'VC precip + ta')
plt.plot(inp_dt1_rad_ta, inp_swe1_VC_rad_ta,linewidth = 1.5, color = 'darkslategrey', label = 'VC rad + ta')
plt.plot(inp_dt1_precip_rad, inp_swe1_VC_precip_rad,linewidth = 2,  color = 'teal', label = 'VC precip + rad')
plt.plot(inp_dt1_ta, inp_swe1_VC_ta,linewidth = 3, color = 'limegreen', label = 'VC ta')
plt.plot(inp_dt1_rad, inp_swe1_VC_rad,linewidth = 3, color = 'forestgreen', label = 'VC rad')
plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VC_precip_rad_ta,linewidth = 4, color = 'lightseagreen',label = 'VC precip + rad + ta')


plt.plot(inp_dt1_precip_ta, inp_swe1_VN_precip_ta, '--',linewidth = 1, color = 'violet', label = 'VN precip + ta')
plt.plot(inp_dt1_rad_ta, inp_swe1_VN_rad_ta,'--',linewidth = 1.5,color = 'hotpink', label = 'VN rad + ta')
plt.plot(inp_dt1_precip_rad, inp_swe1_VN_precip_rad,'--',linewidth = 2,color = 'darkmagenta', label = 'VN precip + rad')
plt.plot(inp_dt1_ta, inp_swe1_VN_ta,'--',linewidth = 3,color = 'lightcoral', label = 'VN ta')
plt.plot(inp_dt1_rad, inp_swe1_VN_rad,'--',linewidth = 3,color = 'mediumpurple', label = 'VN rad')
plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VN_precip_rad_ta, '--',linewidth = 4,color = 'magenta', label = 'VN precip + rad + ta')

plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month

monthsFmt = mdates.DateFormatter('%b')

ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylabel(r"SWE mm w.e.")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
plt.xlabel('Month')


ax1 =  plt.subplot(812, sharex=ax0)
plt.title('dsc_snow-param albedo, Station : {}, Year : {}'.format(Stname[0], year))
plt.plot(dtobs_year, datobs_year, 'o', label = 'Observed')
plt.plot(inp_dt1_obs1, inp_swe1_obs1,'*', label = 'Obs precip')
plt.plot(inp_dt1_precip, inp_swe1_precip, '.', color = 'firebrick', label = 'VCSN precip')

plt.plot(inp_dt1_precip_ta, inp_swe1_VC_precip_ta, linewidth = 1, color = 'seagreen', label = 'VC precip + ta')
plt.plot(inp_dt1_rad_ta, inp_swe1_VC_rad_ta,linewidth = 1.5, color = 'darkslategrey', label = 'VC rad + ta')
plt.plot(inp_dt1_precip_rad, inp_swe1_VC_precip_rad,linewidth = 2,  color = 'teal', label = 'VC precip + rad')
plt.plot(inp_dt1_ta, inp_swe1_VC_ta,linewidth = 3, color = 'limegreen', label = 'VC ta')
plt.plot(inp_dt1_rad, inp_swe1_VC_rad,linewidth = 3, color = 'forestgreen', label = 'VC rad')
plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VC_precip_rad_ta,linewidth = 4, color = 'lightseagreen',label = 'VC precip + rad + ta')


plt.plot(inp_dt1_precip_ta, inp_swe1_VN_precip_ta, '--',linewidth = 1, color = 'violet', label = 'VN precip + ta')
plt.plot(inp_dt1_rad_ta, inp_swe1_VN_rad_ta,'--',linewidth = 1.5,color = 'hotpink', label = 'VN rad + ta')
plt.plot(inp_dt1_precip_rad, inp_swe1_VN_precip_rad,'--',linewidth = 2,color = 'darkmagenta', label = 'VN precip + rad')
plt.plot(inp_dt1_ta, inp_swe1_VN_ta,'--',linewidth = 3,color = 'lightcoral', label = 'VN ta')
plt.plot(inp_dt1_rad, inp_swe1_VN_rad,'--',linewidth = 3,color = 'mediumpurple', label = 'VN rad')
plt.plot(inp_dt1_precip_rad_ta, inp_swe1_VN_precip_rad_ta, '--',linewidth = 4,color = 'magenta', label = 'VN precip + rad + ta')

plt.gcf().autofmt_xdate()
months = mdates.MonthLocator()  # every month

monthsFmt = mdates.DateFormatter('%b')

ax = plt.gca()
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylabel(r"SWE mm w.e.")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
plt.xlabel('Month')

ax2 = plt.subplot(813, sharex=ax1)
ax3 = plt.subplot(814, sharex=ax2)
ax4 = plt.subplot(815, sharex=ax3)
ax5 = plt.subplot(816, sharex=ax4)
ax6 = plt.subplot(817, sharex=ax5)
ax7 = plt.subplot(818, sharex=ax6)

plt.legend()


manager = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.legend()
plt.show()
plt.close()