# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml

# compatibility for windows or linux
projects = '/shareddrive/Projects' #
temp = '/mnt/data2'

path_to_ddf_run = "/home/jono/dsc_snow/ddf/ddf_run"
path_to_namelist = projects + '/Projects/DSC-Snow/runs/input/nevis_2D_test_erebus/ddf_config24.txt'

years = range(2016, 2016+1)

for year in years:

    # read in namelist
    nml = f90nml.read(path_to_namelist)
    # modify for given run
    nml['ddf_config']['ClimateSource'] = projects + '/DSC-Snow/input_data_hourly/met_inp_Nevis_nztm250m_{}.nc'.format(year)
    nml['ddf_config']['OutputFile'] = projects + '/DSC-Snow/runs/output/nevis_2D_test_erebus/Nevis_nztm250m_{}_test24.nc'.format(year)
    nml['ddf_config']['starttime'][0] = year
    nml['ddf_config']['endtime'][0] = year + 1
    # write namelist
    path_to_output_namelist = projects + '/DSC-Snow/runs/input/nevis_2D_test_erebus/ddf_config_{}_test24.txt'.format(year)
    nml.write(path_to_output_namelist, force=True)

    # # copy the met input onto P drive
    # metinfile = nml['ddf_config']['ClimateSource']
    # shutil.copy(temp + '/DSC-Snow/input_data_hourly/' + metinfile.split('/')[-1], metinfile)

    # run model
    args = [path_to_ddf_run, path_to_output_namelist]
    subprocess.call(args)

    # # move output file to Temp drive
    # outfile = nml['ddf_config']['OutputFile']
    # shutil.move(outfile, temp + '/DSC-Snow/runs/output/nevis_2D_test_erebus/' + outfile.split('/')[-1])
    #
    # # remove the met input from P drive
    # os.remove(metinfile)
