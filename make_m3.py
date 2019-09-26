#Author: Peirong Lin (2018-09; uploaded in 2019-09)
#This script takes the inputs: (1) daily runoff data (runoff.nc), (2) weight_table_xxx.csv generated from make_weight.py, (3) basin_id.csv, and generates outputs m3_riv_xxx.nc (which is written in the format of the RAPID Vlat input file format
#Parallel computing can be done through MPI

import pandas as pd
import glob
from netCDF4 import Dataset
import numpy as np
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read runoff dataset
r = Dataset('runoff.nc')
ntime = 12784 #User need to define time time here
nlon = 1440
print(ntime)

#create m3_riv file for each Pfafstetter Level_01 basin
for i in range(1,9)[rank::size]:
    print('******* '+str(i)+' **********')
    #read weight table
    df1 = pd.read_csv('tables/weight_table_pfaf_%02d'%i+'.csv')
    #read RAPID basin_id and merge with each continent's weight table
    df0 = pd.read_csv('tables/basin_id_%02d'%i+'.csv',header=None)
    df0.columns = ['COMID']
    #merge: weight table in the order of BASIN_ID
    df = df0.merge(df1,on='COMID',how='left')
    df.fillna(0,inplace=True)
    df['ix'] = df['ix'].astype(int)
    df['iy'] = df['iy'].astype(int)
    df['number'] = df['number'].astype(int)
    nriver = len(df['COMID'].unique())
    print(nriver)
    df['ix'].loc[df['ix']==nlon] = nlon-1 #tmp fix

    #write to Vlat file
    filename = 'm3_riv/m3_riv_pfaf_%02d'%i+'.nc'
    fon = Dataset(filename, 'w', format='NETCDF3_CLASSIC')  #use NETCDF3!!! import for correct read in RAPID
    Time = fon.createDimension('Time', ntime)
    COMID = fon.createDimension('COMID', nriver)
    m3_riv = fon.createVariable('m3_riv', np.float32,('Time','COMID'))
    fon.close()

    #write HydroID into netcdf
    data = pd.DataFrame(df['COMID'].unique(),columns=['COMID'])

    print('--- WRITING TIME DATA ---')
    fon = Dataset(filename, 'a')
    for itime in range(0,ntime):
        print(itime)
        rr = r.variables['runoff'][itime,:,:]#NOTE: Change this variable name if using another; make sure lat/lon consistent with those set in make_intersect.py and make_weight.py!!!
        #replace NaN value and Infinityf values, if any
        rr[np.isnan(rr)] = 0. 
        rr[rr>10000000.]=0. 
        rr[rr<-10000000.] =0.
        df['runoff']=rr[df['iy'],df['ix']]/1000.*df['area']*1000.*1000.  #convert unit to m3;NOTE FOR UNIT!
        data['m3_riv']=df.groupby('COMID')['runoff'].agg(['sum']).values
        m3_riv[itime,:]=data['m3_riv'].values

    #close netcdf file
    print(m3_riv.shape)
    fon.close() 
    print('**** write to netcdf' +filename+' *****')



