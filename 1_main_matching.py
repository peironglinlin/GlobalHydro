import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from scipy import stats
import math
import time
import sys
import os

############GLOBAL VARIABLES#############################
ds = xr.open_dataset('../../../runoff_mswep_subdaily.nc') #runoff to be corrected
#read lat/lon
lat =  Dataset('../../../runoff_mswep_subdaily.nc').variables['lat'][:]
lon =  Dataset('../../../runoff_mswep_subdaily.nc').variables['lon'][:]
percentiles = [1,5,10,20,50,80,90,95,99]
#read Q characeteristics map (stored in one netcdf)
ds2 = xr.open_dataset('../netcdf/concat_Q_con_all_nomean_new2_bc.nc') #Q characteristics
#folders to store the output files
output_dir = 'output_folder'
output_dir_000 = 'output_folder_000' #store those with zero values
########################################################


def SparseCDFMatching(obs1,obs2,mod1,mod2,original):
    ratio1 = obs1/mod1
    ratio2 = obs2/mod2
    n = len(original)
    if ratio1 > 1000:
        ratio1 = 1000. #use this to avoid very rare cases
    if ratio2 > 1000:
        ratio2 = 1000. #use this to avoid very rare cases
    print('ratio = %.2f'%ratio1,'  %.2f'%ratio2)
    ratioj = [math.exp(math.log(ratio1)+j/n*(math.log(ratio2)-math.log(ratio1))) for j in range(n)]  #assume log-linear CDF between percentiles (Q characteristics)
    corr = [original[j]*math.exp(math.log(ratio1)+j/n*(math.log(ratio2)-math.log(ratio1))) for j in range(n)]
    return np.array(corr)

def BiasCorrectPercentiles(rmod,rmodsort,robssort,mod_quans,pct,cdf):
    rmodcorr = np.zeros(len(rmod)) #initialize corrected modelled runoff
    for i in range(1,len(percentiles)):
        pct1 = percentiles[i-1]
        pct2 = percentiles[i]
        print('--- Percentile '+str(pct1)+' to '+str(pct2)+' ----')
        obs1 = robssort[i-1]
        obs2 = robssort[i]
        mod1 = mod_quans[i-1]
        mod2 = mod_quans[i]
        print('************************')
        print('obs1=%.6f'%obs1,'mod1=%.6f'%mod1)
        print('obs2=%.6f'%obs2,'mod2=%.6f'%mod2)
        rmodnow = rmodsort[(cdf<pct2) & (cdf>=pct1)]
        rmodcorr[(cdf<pct2) & (cdf>=pct1)] = SparseCDFMatching(obs1,obs2,mod1,mod2,rmodnow)
    #extrapolate below
    ratio99 = robssort[0]/mod_quans[0]
    rmodcorr[cdf<percentiles[0]] = [ratio99*x for x in rmodsort[cdf<percentiles[0]]] 
    #extrapolate above
    npctl = len(percentiles)
    import pdb;pdb.set_trace()
    ratio1 = robssort[npctl-1]/mod_quans[npctl-1]
    rmodcorr[cdf>=percentiles[npctl-1]] = [ratio1*x for x in rmodsort[cdf>=percentiles[npctl-1]]] #order follows CDF
    import pdb;pdb.set_trace()
    r2 = [stats.scoreatpercentile(list(rmodcorr), x) for x in pct]
    return r2

def CorrectPixel(iy,ix):
    print(iy,ix)
    start = time.time()
    #read time series data
    tmp = ds.sel(lon=lon[ix], lat=lat[iy], method='nearest')
    rmod = tmp['totrun']  #final unit mm (convert from m to mm)
    #read Q characteristic data
    robs = ds2.sel(lon=lon[ix],lat=lat[iy],method='nearest')['Band1'].values
    if rmod.isnull().any():  #if model has NaN value:
        fon = output_dir_000+'/runoff_%04d'%iy+'_%04d'%ix+'.csv'
        pd.DataFrame({'r_mod':rmod,'pct':[0.],'r_corr':rmod}).to_csv(fon,index=False)
    else:
        if np.mean(robs) == 0.:
            fon = output_dir_000+'/runoff_%04d'%iy+'_%04d'%ix+'.csv'
            pd.DataFrame({'r_mod':[0.],'pct':[0.],'r_corr':[0.]}).to_csv(fon,index=False)
        else:
            #####  !!!!!need to change rmodlist (this procedure is slow if using xarray
            rmodlist = list(rmod.values)    
            #calculate percentiles for constructing CDF
            pct = [stats.percentileofscore(rmodlist, x, kind='weak') for x in rmodlist]
            #sort original data and calculated percentiles
            df = pd.DataFrame({'rmod':rmod,'pct':pct})
            df.sort_values(by='pct',inplace=True) #sort data (quicksort)
            cdf = df['pct'].values
            rmodsort = df['rmod'].values
            mod_quans = [stats.scoreatpercentile(rmodlist,x) for x in percentiles]
            robssort = np.sort(robs)
            #call bias_correction algorithm
            r_corr = BiasCorrectPercentiles(rmod,rmodsort,robssort,mod_quans,pct,cdf)
            import pdb;pdb.set_trace()
            fon = output_dir+'/runoff_%04d'%iy+'_%04d'%ix+'.csv'
            pd.DataFrame({'r_mod':rmod.values,'pct':pct,'r_corr':r_corr}).to_csv(fon,index=False)
        print('*** TOTAL TIME SPENT = %s seconds ***'%(time.time()-start))

    
################ MAIN PROGRAM ###############################    
import os
import pickle
fin = open('pixels.pickle','rb')  #pixels.pickle stored all ~0.24 million cell locations that need BC
print(len(pixels))

tmp = pickle.load(fin)
interval = math.ceil(len(tmp)/njob)
print('*** TOTAL = %s PICKLE POINTS ***'%len(tmp))
pixels = tmp[ijob*interval:(ijob+1)*interval] #self-made parallel processing to speed up BC procedure (by control.py)
print('----- now processing '+str(ijob*interval)+' ~ '+str((ijob+1)*interval)+' ------------')

for i in range(len(pixels))[rank::size]: #MPI used here to speed up BC procedure
    if not os.path.isfile(output_dir+'/runoff_%04d'%pixels[i][0]+'_%04d'%pixels[i][1]+'.csv'): #if already corrected; skip
        print('******* lat=%.3f'%lat[pixels[i][0]]+' ; lon=%.3f'%lon[pixels[i][1]]+' *********')
        CorrectPixel(pixels[i][0],pixels[i][1])
        
