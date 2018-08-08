# script to run the Fortran version of dsc_snow model and iterate through different namelist options, file names etc

import subprocess
import os
import shutil
import f90nml

# compatibility for windows or linux
projects = "P:" #'/shareddrive' #
temp = "T:" # '/mnt/temp'

path_to_ddf_run = "/home/jono/dsc_snow/ddf/ddf_run"
path_to_namelist = projects + '/Projects/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_jobst_ucc_4.txt'

years = range(2000,2016+1)

for year in years:

    # read in namelist
    nml = f90nml.read(path_to_namelist)
    # modify for given run
    nml['ddf_config']['ClimateSource'] = projects + '/Projects/DSC-Snow/input_data_hourly/met_inp_Clutha_nztm250m_{}_jobst_ucc.nc'.format(year)
    nml['ddf_config']['OutputFile'] = projects + '/Projects/DSC-Snow/runs/output/clutha_nztm250m_erebus/Clutha_nztm250m_{}_jobst_ucc_4_test.nc'.format(year)
    nml['ddf_config']['starttime'][0] = year
    nml['ddf_config']['endtime'][0] = year + 1
    # write namelist
    path_to_output_namelist = projects + '/Projects/DSC-Snow/runs/input/clutha_nztm250m_erebus/ddf_config_{}_jobst_ucc_4_test.txt'.format(year)
    nml.write(path_to_output_namelist)

    # copy the met input onto P drive
    metinfile = nml['ddf_config']['ClimateSource']
    shutil.copy(temp + '/DSC-Snow/input_data_hourly/' + metinfile.split('/')[-1], metinfile)

    # run model
    args = [path_to_ddf_run, path_to_output_namelist]
    subprocess.call(args)

    # move output file to Temp drive
    outfile = nml['ddf_config']['OutputFile']
    shutil.move(outfile, temp + '/DSC-Snow/runs/output/clutha_nztm250m_erebus/' + outfile.split('/')[-1])

    # remove the met input from P drive
    os.remove(metinfile)