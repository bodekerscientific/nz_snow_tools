import numpy as np
from nz_snow_tools.util.utils import make_regular_timeseries
import matplotlib.pylab as plt
import datetime as dt

# load maxmin temperature data
inp_dat = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_MaxMinTemp.txt", delimiter=',',
                        skip_header=9, skip_footer=8)
inp_time = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_MaxMinTemp.txt", usecols=(1),
                         dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time])

ta = inp_dat[:, 8]
rh = inp_dat[:, 9]

# load radiation data
inp_dat1 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Radiation.txt", delimiter=',',
                         skip_header=9, skip_footer=14)
inp_time1 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Radiation.txt", usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=14)
inp_dt1 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time1])

radiation = inp_dat1[:, 2]*1e6/3600

# load precipitation data
inp_dat2 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Rain.txt", delimiter=',',
                         skip_header=9, skip_footer=8)
inp_time2 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Rain.txt", usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt2 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time2])

precip = inp_dat2[:, 2]

# load wind speed data
inp_dat3 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Wind.txt", delimiter=',',
                         skip_header=9, skip_footer=8)
inp_time3 = np.genfromtxt("C:/Users/Bonnamourar/Desktop/SIN/Mueller/Mueller_2014-2019_Wind.txt", usecols=(1),
                          dtype=(str), delimiter=',', skip_header=9, skip_footer=8)
inp_dt3 = np.asarray([dt.datetime.strptime(t, '%Y%m%d:%H%M') for t in inp_time3])

wind = inp_dat3[:, 3]




ax1 = plt.subplot(511)
plt.plot(inp_dt,ta,label = "Temperature", color = "red")
plt.ylabel('Temperature (C)')
plt.title("Mueller")

ax2 = plt.subplot(512, sharex=ax1)
plt.plot(inp_dt3,wind, label = "Wind Speed", color = "yellow")
plt.ylabel('Wind Speed (m/s)')


ax3 = plt.subplot(513, sharex=ax2)
plt.plot(inp_dt2, np.cumsum(precip), label="Precipitation")
plt.ylabel('cummulative Precipitation (mm)')


ax4 = plt.subplot(514, sharex=ax3)
plt.plot(inp_dt1,radiation,label = "Radiation", color = "orange")
plt.ylabel('Radiation (MJ/m2)')

ax5 = plt.subplot(515, sharex=ax4)
plt.plot(inp_dt,rh,label = "Relative humidity", color = "black")
plt.ylabel('Rh (%)')


plt.show()


# load each year

for i in range(0,13):
    try:
        start_t = np.where(inp_dt == dt.datetime(2007+i,4,1,00,00))[0][0]
        end_t = np.where(inp_dt == dt.datetime(2008+i,4,1,00,00))[0][0]
        start_t1 = np.where(inp_dt1 == dt.datetime(2007+i,4,1,00,00))[0][0]
        end_t1 = np.where(inp_dt1 == dt.datetime(2008+i,4,1,00,00))[0][0]
        start_t2 = np.where(inp_dt2 == dt.datetime(2007+i,4,1,00,00))[0][0]
        end_t2 = np.where(inp_dt2 == dt.datetime(2008+i,4,1,00,00))[0][0]
        start_t3 = np.where(inp_dt3 == dt.datetime(2007+i,4,1,00,00))[0][0]
        end_t3 = np.where(inp_dt3 == dt.datetime(2008+i,4,1,00,00))[0][0]
    #assert inp_dt[start_t:end_t]==inp_dt1[start_t1:end_t1]==inp_dt2[start_t2:end_t2]==inp_dt3[start_t3:end_t3]
        print(np.all(inp_dt[start_t:end_t]==inp_dt1[start_t1:end_t1]))
        print(np.all(inp_dt[start_t:end_t]==inp_dt2[start_t2:end_t2]))
        print(inp_dt[start_t:end_t].shape)
        print(inp_dt1[start_t1:end_t1].shape)
        print(inp_dt2[start_t2:end_t2].shape)
        output = np.transpose(np.vstack((inp_dt[start_t:end_t],rh[start_t:end_t],ta[start_t:end_t],radiation[start_t1:end_t1],precip[start_t2:end_t2])))
        np.save("C:/Users/Bonnamourar/OneDrive - NIWA/Station data/Mueller_{}".format(2007+i),output)
    except:
        print("Missing data for {}".format(2007+i))




#plt.plot(t, s2)
# make these tick labels invisible
#plt.setp(ax2.get_xticklabels(), visible=False)


inp_dt = make_regular_timeseries(dt.datetime(2010, 10, 25, 00, 30), dt.datetime(2012, 9, 2, 00, 00), 1800)

start_t = 9600 - 1  # 9456 = start of doy 130 10th May 2011 9600 = end of 13th May,18432 = start of 11th Nov 2013,19296 = 1st december 2011
end_t = 21360  # 20783 = end of doy 365, 21264 = end of 10th January 2012, 21360 = end of 12th Jan
inp_doy = inp_dat[start_t:end_t, 2]
inp_hourdec = inp_dat[start_t:end_t, 3]
# make grids of input data
grid_size = 1
grid_id = np.arange(grid_size)
inp_ta = inp_dat[start_t:end_t, 8][:, np.newaxis] * np.ones(grid_size) + 273.16  # 2 metre air temp in C
inp_precip = inp_dat[start_t:end_t, 21][:, np.newaxis] * np.ones(grid_size)  # precip in mm
inp_sw = inp_dat[start_t:end_t, 15][:, np.newaxis] * np.ones(grid_size)
inp_sfc = inp_dat[start_t - 1:end_t, 19]  # surface height change
inp_sfc -= inp_sfc[0]  # reset to 0 at beginning of period
