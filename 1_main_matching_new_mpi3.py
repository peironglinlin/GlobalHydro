import sys, os, pickle
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

############GLOBAL VARIABLES#############################
#read lat/lon point
if rank==0:
    ds = xr.open_dataset(sys.argv[1]) #runoff to be corrected
    ds.load()

ftpl = Dataset(sys.argv[1])
lat =  ftpl.variables['lat'][:]
lon =  ftpl.variables['lon'][:]
ntim = ftpl.variables['time'].shape[0]
buffsiz = ntim*10

#read reference characteristics maps
percentiles = [1,5,10,20,50,80,90,95,99]
ds2 = xr.open_dataset('concat_Q_con_all_nomean_new.nc') #Q characteristics
ds2.load()

#read land-sea mask data
dsmask = xr.open_dataset('landsea_mask_new.nc')
dsmask.load()

#user-defined
varname = 'runoff'
output_dir = 'corrected'
output_dir_000 = 'corrected'
########################################################


def SparseCDFMatching(obs1,obs2,mod1,mod2,original):
    ratio1 = obs1/mod1
    ratio2 = obs2/mod2
    n = len(original)
    oranks = stats.rankdata(original, 'min').astype(np.float32)
    ormax  = np.max(oranks)+1
    if ratio1 > 1000:
        ratio1 = 1000. #use this to avoid very rare cases
    if ratio2 > 1000:
        ratio2 = 1000. #use this to avoid very rare cases
    print('ratio = %.2f'%ratio1,'  %.2f'%ratio2)
    #ratioj = [math.exp(math.log(ratio1)+oranks[j]/ormax*(math.log(ratio2)-math.log(ratio1))) for j in range(n)]  #assume log-linear CDF between percentiles (Q characteristics)
    corr = [original[j]*math.exp(math.log(ratio1)+oranks[j]/ormax*(math.log(ratio2)-math.log(ratio1))) for j in range(n)]
    return np.array(corr) #,np.array(ratioj)

def BiasCorrectPercentiles(rmod,robssort,mod_quans,pct):
    rmodcorr = np.zeros(len(rmod)) #initialize corrected modelled runoff
#     r_ratio = np.zeros(len(rmod)) #initialize corrected modelled runoff
    for i in range(1,len(percentiles)):
        pct1 = percentiles[i-1]
        pct2 = percentiles[i]
#         print('--- Percentile '+str(pct1)+' to '+str(pct2)+' ----')
        obs1 = robssort[i-1]
        obs2 = robssort[i]
        mod1 = mod_quans[i-1]
        mod2 = mod_quans[i]
#         print('************************')
#         print('obs1=%.6f'%obs1,'mod1=%.6f'%mod1)
#         print('obs2=%.6f'%obs2,'mod2=%.6f'%mod2)
        rmodnow = rmod[(pct<pct2) & (pct>=pct1)]
        if rmodnow.size>0:
            rmodcorr[(pct<pct2) & (pct>=pct1)] = SparseCDFMatching(obs1,obs2,mod1,mod2,rmodnow)
    #extrapolate below
    rmodcorr[pct<percentiles[0]] = rmod[pct<percentiles[0]]*(robssort[0]/mod_quans[0])
    #extrapolate above
    npctl = len(percentiles)
    rmodcorr[pct>=percentiles[npctl-1]] = rmod[pct>=percentiles[npctl-1]]*(robssort[npctl-1]/mod_quans[npctl-1])

    return rmodcorr #,r_ratio_final

def CorrectPixel(rmod, robs):
    #start = time.time()
    if np.isnan(rmod).any():  #if model has NaN value:
        result = pd.DataFrame({'r_mod':[0.],'pct':[0.],'r_corr':[0.],'r_ratio1':[1.]})
    else:
        if np.mean(robs) == 0.:
            result = pd.DataFrame({'r_mod':[0.],'pct':[0.],'r_corr':[0.],'r_ratio1':[1.]})
        else:
            #####  !!!!!need to change rmodlist (this procedure is slow if using xarray
            rmodlist = list(rmod)    
            #calculate percentiles for constructing CDF
            pct = stats.rankdata(rmodlist, 'max')*100.0/len(rmodlist)
            mod_quans = [stats.scoreatpercentile(rmodlist,x) for x in percentiles]
            robssort = np.sort(robs)
            #call bias_correction algorithm
            r_corr = BiasCorrectPercentiles(rmod,robssort,mod_quans,pct)
            r_ratio1 = r_corr/rmod
            result = pd.DataFrame({'r_mod':rmod,'pct':pct,'r_corr':r_corr,'r_ratio1':r_ratio1})
        #print('*** TOTAL TIME SPENT = %s seconds ***'%(time.time()-start))

    return result


################ MAIN PROGRAM ###############################    
fin = open('pixels.pickle','rb')
pixels = pickle.load(fin)

print(len(pixels))

if rank==0:

    filename = sys.argv[2]
    print('... writing to '+filename+' ...')
    fon = Dataset(filename, 'w', format='NETCDF4')  #use NETCDF3!!! import for correct read in RAPID

    fon.createDimension('time', None)
    tvar = fon.createVariable('time', 'f4',('time',))
    tvar.units = ftpl.variables['time'].units
    tvar.long_name = ftpl.variables['time'].long_name
    tvar[:] = ftpl.variables['time'][:]

    nlat = ftpl.variables['lat'].shape[0]
    fon.createDimension('lat', nlat)
    latvar = fon.createVariable('lat','f4',('lat'))
    latvar.units = ftpl.variables['lat'].units
    latvar.long_name = ftpl.variables['lat'].long_name
    latvar[:] = ftpl.variables['lat'][:]

    nlon = ftpl.variables['lon'].shape[0]
    fon.createDimension('lon', nlon)
    lonvar = fon.createVariable('lon','f4',('lon'))
    lonvar.units = ftpl.variables['lon'].units
    lonvar.long_name = ftpl.variables['lon'].long_name
    lonvar[:] = ftpl.variables['lon'][:]

    var = fon.createVariable('ratio', 'f4',('time','lat','lon'))
    dvar = np.ones((ntim, nlat, nlon), dtype=np.float32)

cnt = 0
for i in range(len(pixels))[rank::size]:

    cnt = cnt+1

    iy = pixels[i][0]
    ix = pixels[i][1]
    loc = np.array([ix, iy], dtype=np.int32)

    print('******* lat=%.3f'%lat[iy]+' ; lon=%.3f'%lon[ix]+' *********')

    #read time series data
    if rank==0:

        tmp = ds.sel(lon=lon[ix], lat=lat[iy], method='nearest')
        rmod = tmp[varname].values #*1000.  #final unit mm (convert from m to mm)

        if i+size<len(pixels)-1 or len(pixels)%size==0:
            sranks = range(1, size)
        else:
            sranks = range(1, len(pixels)%size)

        for srank in sranks:
            # request location from process srank
            sloc = np.empty(2, dtype=np.int32)
            req3 = comm.irecv(source=srank, tag=cnt*10+3)
            sloc = req3.wait()
            #comm.Recv(sloc, source=srank, tag=cnt*10+3)
            six = sloc[0]
            siy = sloc[1]
            # extract data for the location
            stmp = ds.sel(lon=lon[six], lat=lat[siy], method='nearest')
            srmod = stmp[varname].values.astype(np.float32) #*1000.  #final unit mm (convert from m to mm)
            # send it back to process srank
            req4 = comm.isend(srmod, dest=srank, tag=cnt*10+4)
            req4.wait()
            #comm.Send(srmod, dest=srank, tag=cnt*10+4)

    else:
        # send location to process 0
        req3 = comm.isend(loc, dest=0, tag=cnt*10+3)
        req3.wait()
        #comm.Send(loc, dest=0, tag=cnt*10+3)
        # request rmod data from process 0
        rmod = np.empty(ntim, dtype=np.float32)
        #print('receiving size: %d' % rmod.shape[0])
        req4 = comm.irecv(buffsiz, source=0, tag=cnt*10+4)
        rmod = req4.wait()
        #comm.Recv(rmod, source=0, tag=cnt*10+4)

    #read Q characteristic data
    robs = ds2.sel(lon=lon[ix],lat=lat[iy],method='nearest')['Band1'].values

    df = CorrectPixel(rmod, robs)

    data = df['r_ratio1'].values.astype(np.float32)
    data[np.isinf(data)] = 1.
    data[np.isnan(data)] = 1.
    data[data>10000] = 10000
    if np.mean(df['r_mod']) == 0.:
        print('    ... zero ...')
        data[:] = 1.
    elif df[['r_ratio1']].isnull().values.any(): #if no corrected values are generated
        print('    ... no corrected values generated ...')
        data[:] = 1.
    else:
        print('    ... good ...')

    if rank==0:
        dvar[:,iy,ix] = data

        if i+size<len(pixels)-1 or len(pixels)%size==0:
            sranks = range(1, size)
        else:
            sranks = range(1, len(pixels)%size)

        for srank in sranks:
            # request location from process srank
            sloc = np.empty(2, dtype=np.int32)
            req1 = comm.irecv(source=srank, tag=cnt*10+1)
            sloc = req1.wait()
            # request result data from process srank
            sdata = np.empty(ntim, dtype=np.float32)
            req2 = comm.irecv(buffsiz, source=srank, tag=cnt*10+2)
            sdata = req2.wait()
            # write to file
            six = sloc[0]
            siy = sloc[1]
            dvar[:, siy, six] = sdata

    else:
        # send location to process 0
        req1 = comm.isend(loc, dest=0, tag=cnt*10+1)
        req1.wait()
        # send result data to process 0
        req2 = comm.isend(data, dest=0, tag=cnt*10+2)
        req2.wait()

if rank==0:
    print('Writing to file')
    var[:] = dvar
    fon.close()

        
