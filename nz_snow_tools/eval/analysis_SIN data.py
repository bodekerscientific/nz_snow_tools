import numpy as np
from nz_snow_tools.snow.clark2009_snow_model import snow_main_simple
from nz_snow_tools.util.utils import make_regular_timeseries,convert_datetime_julian_day,convert_dt_to_hourdec,nash_sut, mean_bias, rmsd, mean_absolute_error
import matplotlib.pylab as plt
import datetime as dt
import matplotlib.dates as mdates
from nz_snow_tools.util.utils import fill_timeseries

#CASTLE MOUNT Clark2009 and Albedo
# C2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_2012.npy"
# C2013_file = "C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_2013.npy"
# C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_2014.npy"
# C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_2015.npy"
# C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_clark2009_2016.npy"
# A2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_2012.npy"
# A2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_2013.npy"
# A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_2014.npy"
# A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_2015.npy"
# A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Castle Mount/CastleMount_npy files/Castle Mount_dsc_snow-param albedo_2016.npy"
# LARKINS Clark2009 and Albedo
# C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_2014.npy"
# C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_2015.npy"
# C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_2016.npy"
# C2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_2017.npy"
# C2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_clark2009_2018.npy"
# A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_2014.npy"
# A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_2015.npy"
# A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_2016.npy"
# A2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_2017.npy"
# A2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Larkins/Larkins_npy files/Larkins_dsc_snow-param albedo_2018.npy"
# MAHANGA Clark2009 and Albedo
# C2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2009.npy"
# C2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2010.npy"
# C2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2011.npy"
# C2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2012.npy"
# C2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2013.npy"
# C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2014.npy"
# C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2015.npy"
# C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2016.npy"
# C2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2017.npy"
# C2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_clark2009_2018.npy"
# A2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2009.npy"
# A2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2010.npy"
# A2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2011.npy"
# A2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2012.npy"
# A2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2013.npy"
# A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2014.npy"
# A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2015.npy"
# A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2016.npy"
# A2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2017.npy"
# A2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mahanga/Mahanga_npy files/Mahanga_dsc_snow-param albedo_2018.npy"
# MUELLER Clark2009 and Albedo
C2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2011.npy"
C2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2012.npy"
C2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2013.npy"
C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2014.npy"
C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2015.npy"
C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2016.npy"
C2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2017.npy"
C2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_clark2009_2018.npy"
A2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2011.npy"
A2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2012.npy"
A2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2013.npy"
A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2014.npy"
A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2015.npy"
A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2016.npy"
A2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2017.npy"
A2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Mueller/Mueller_npy files/Mueller_dsc_snow-param albedo_2018.npy"
# MURCHISON Clark2009 and Albedo
# C2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2009.npy"
# C2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2010.npy"
# C2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2011.npy"
# C2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2012.npy"
# C2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2013.npy"
# C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2014.npy"
# C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2015.npy"
# C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2016.npy"
# C2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2017.npy"
# C2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_clark2009_2018.npy"
# A2009_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2009.npy"
# A2010_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2010.npy"
# A2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2011.npy"
# A2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2012.npy"
# A2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2013.npy"
# A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2014.npy"
# A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2015.npy"
# A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2016.npy"
# A2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2017.npy"
# A2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Murchison/Murchison_npy files/Murchison_dsc_snow-param albedo_2018.npy"
# PHILISTINE Clark2009 and Albedo
# C2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2011.npy"
# C2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2012.npy"
# C2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2013.npy"
# C2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2014.npy"
# C2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2015.npy"
# C2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2016.npy"
# C2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2017.npy"
# C2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_clark2009_2018.npy"
# A2011_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2011.npy"
# A2012_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2012.npy"
# A2013_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2013.npy"
# A2014_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2014.npy"
# A2015_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2015.npy"
# A2016_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2016.npy"
# A2017_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2017.npy"
# A2018_file ="C:/Users/Bonnamourar/OneDrive - NIWA/SIN calibration timeseries/Philistine/Philistine_npy files/Philistine_dsc_snow-param albedo_2018.npy"

# load clark2009 model
inp_clark = np.load(C2015_file,allow_pickle=True)
inp_time1 = inp_clark[:,0]
inp_swe1 = np.asarray(inp_clark[:,1],dtype=np.float)
plot_dt = inp_clark[:, 0] # model stores initial state

plt.plot(inp_time1, inp_swe1, color ="red")
plt.show()


# load dsc_param_albedo model
inp_albedo = np.load(A2015_file,allow_pickle=True)
inp_time2 = inp_albedo[:,0]
inp_swe2 = np.asarray(inp_albedo[:,1],dtype=np.float)

plt.plot(inp_time2, inp_swe2, color ="aquamarine")
plt.show()

# MUELLER SWE csv file
csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mueller SWE.csv"
# MAHANGA SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Mahanga SWE.csv"
# LARKINS SWE csv file
# csv_file ="C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Larkins SWE.csv"
# CASTLE MOUNT SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Castle Mount SWE.csv"
# MURCHISON SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Murchison SWE.csv"
# PHILISTINE SWE csv file
# csv_file = "C:/Users/Bonnamourar/OneDrive - NIWA/CSV SWE/Philistine SWE.csv"

# load observed data
inp_datobs = np.genfromtxt(csv_file, delimiter=',',usecols=(1),
                        skip_header=4)*1000
inp_timeobs = np.genfromtxt(csv_file, usecols=(0),
                         dtype=(str), delimiter=',', skip_header=4)
inp_dtobs = np.asarray([dt.datetime.strptime(t, '%d/%m/%Y %H:%M') for t in inp_timeobs])
ind = np.logical_and(inp_dtobs >= plot_dt[0],inp_dtobs <= plot_dt[-1])

inp_dtobs_clean, inp_datobs_clean = fill_timeseries(inp_dtobs[ind], inp_datobs[ind], 3600)

plt.plot(inp_dtobs_clean,inp_datobs_clean,"o", label = "Observed SWE", color = "blue")
plt.show()

#mean = mean_bias(inp_swe1,inp_datobs[ind])

