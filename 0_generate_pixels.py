#Peirong Lin (uploaded 2019-03-17)
#This program generates all lat/lon for grid cells that need BC
#For this demonstration practice, there are ~0.24 million cells

import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from scipy import stats
import math
import time
from mpi4py import MPI

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read land-sea mask data
dsmask = xr.open_dataset('landsea_mask_new2_bc.nc')
    
################ MAIN PROGRAM ###############################    
# for ix in range(0,len(lon)) for iy in range(0,len(lat))[rank::size]:
pixels = []
import os
print('---- SCREENING PIXELS FOR LAND ONLY ----')
for iy in range(0,600)[rank::size]:
    for ix in range(0,1440)[rank::size]:
        if (dsmask.sel(lon=lon[ix], lat=lat[iy], method='nearest')['mask'].values == 1): # & (not os.path.isfile('corrected_vic/runoff_%04d'%iy+'_%04d'%ix+'.csv')):
            print(iy,ix)
            pixels.append((iy,ix))  #only correct land pixels
print(len(pixels))

import pickle
fon = open('pixels.pickle','wb')
pickle.dump(pixels,fon)
fon.close()
print('.... writing to pixel pickle file ...')

        
