import numpy as np
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt

#MUELLER files
ta_file = "C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2007-2019/Mueller_2007-2019_MaxMinTemp.txt"
radiation_mueller_file = "C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2007-2019/Mueller_2007-2019_Radiation.txt"
precipitation__mueller_file = "C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2007-2019/Mueller_2007-2019_Rain.txt"
wind_mueller_file = "C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2007-2019/Mueller_2007-2019_Wind.txt"
save_mueller_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller/Mueller_{}"

#CASTLE MOUNT files
#ta_file = "C:/Users/Bonnamourar/Desktop/SIN/Castle Mount/Castel Mount_2007-2019/CastleMt_2007-2019_MaxMinTemp.txt"
#radiation_cstmount_file = "C:/Users/Bonnamourar/Desktop/SIN/Castle Mount/Castel Mount_2007-2019/CastleMt_2007-2019_Radiation.txt"
#precipitation__cstmount_file = "C:/Users/Bonnamourar/Desktop/SIN/Castle Mount/Castel Mount_2007-2019/CastleMt_2007-2019_Rain.txt"
#wind_cstmount_file = "C:/Users/Bonnamourar/Desktop/SIN/Castle Mount/Castel Mount_2007-2019/CastleMt_2007-2019_Wind.txt"
#save_cstmount_file ="C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Castle Mount/CastleMount_{}"
#change the name
# load maxmin temperature data
inp_dat = np.genfromtxt(ta_file, delimiter=',',
                        skip_header=9, skip_footer=8)
inp_time = np.genfromtxt(ta_file, usecols=(1),
                         dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time])

ta = inp_dat[:, 8]
rh = inp_dat[:, 9]

# load radiation data
inp_dat1 = np.genfromtxt(radiation_mueller_file, delimiter=',',
                         skip_header=9, skip_footer=8)
inp_time1 = np.genfromtxt(radiation_mueller_file, usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=9)
inp_dt1 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time1])

radiation = inp_dat1[:, 2]*1e6/3600

# load precipitation data
inp_dat2 = np.genfromtxt(precipitation__mueller_file, delimiter=',',
                         skip_header=9, skip_footer=8)
inp_time2 = np.genfromtxt(precipitation__mueller_file, usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt2 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time2])

precip = inp_dat2[:, 2]

# load wind speed data
inp_dat3 = np.genfromtxt(wind_mueller_file, delimiter=',',
                         skip_header=9, skip_footer=8)
inp_time3 = np.genfromtxt(wind_mueller_file, usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt3 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time3])

wind = inp_dat3[:, 3]


# parameter plots

ax1 = plt.subplot(511)
plt.plot(inp_dt,ta,label = "Temperature", color = "darkred")
plt.ylabel('Temperature (C)')
plt.title("Castle Mount")

ax2 = plt.subplot(512, sharex=ax1)
plt.plot(inp_dt3,wind, label = "Wind Speed", color = "aquamarine")
plt.ylabel('Wind Speed (m/s)')


ax3 = plt.subplot(513, sharex=ax2)
plt.plot(inp_dt2, np.cumsum(precip), label="Precipitation", color = "mediumblue")
plt.ylabel('cummulative Precipitation (mm)')


ax4 = plt.subplot(514, sharex=ax3)
plt.plot(inp_dt1,radiation,label = "Radiation", color = "gold")
plt.ylabel('Radiation (W)')

ax5 = plt.subplot(515, sharex=ax4)
plt.plot(inp_dt,rh,label = "Relative humidity", color = "deepskyblue")
plt.ylabel('Rh (%)')


plt.show()


# load each year
for i in range(0,13):
    date0 = 2007 + i
    date1 = 2008 + i
    try:
        start_t = np.where(inp_dt == dt.datetime(date0,4,1,00,00))[0][0]
        end_t = np.where(inp_dt == dt.datetime(date1,4,1,00,00))[0][0]
        start_t1 = np.where(inp_dt1 == dt.datetime(date0,4,1,00,00))[0][0]
        end_t1 = np.where(inp_dt1 == dt.datetime(date1,4,1,00,00))[0][0]
        start_t2 = np.where(inp_dt2 == dt.datetime(date0,4,1,00,00))[0][0]
        end_t2 = np.where(inp_dt2 == dt.datetime(date1,4,1,00,00))[0][0]
        start_t3 = np.where(inp_dt3 == dt.datetime(date0,4,1,00,00))[0][0]
        end_t3 = np.where(inp_dt3 == dt.datetime(date1,4,1,00,00))[0][0]
    #assert inp_dt[start_t:end_t]==inp_dt1[start_t1:end_t1]==inp_dt2[start_t2:end_t2]==inp_dt3[start_t3:end_t3]
        print(np.all(inp_dt[start_t:end_t]==inp_dt1[start_t1:end_t1]))
        print(np.all(inp_dt[start_t:end_t]==inp_dt2[start_t2:end_t2]))
        print(inp_dt[start_t:end_t].shape)
        print(inp_dt1[start_t1:end_t1].shape)
        print(inp_dt2[start_t2:end_t2].shape)
        output = np.transpose(np.vstack((inp_dt[start_t:end_t],rh[start_t:end_t],ta[start_t:end_t],radiation[start_t1:end_t1],precip[start_t2:end_t2])))
        np.save(save_mueller_file.format(date0),output)
    except:
        print("Missing data for {}".format(date0))




