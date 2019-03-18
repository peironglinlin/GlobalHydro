#Peirong Lin (uploaded 2019-03-17)
#This program takes outputs from "make_intersect.py" and organize data input weight table format needed by coupling RAPID with LSMs

import pandas as pd
import glob
from netCDF4 import Dataset
import numpy as np
import os
import geopandas as gpd


for i in range(1,10):
    fin = 'tables/intersect_pfaf_%02d'%i+'.csv'
    print('... read '+fin+' ...')
    f = pd.read_csv(fin)

    #only selected columns
    data=f
    data['number'] = data.groupby('COMID')['COMID'].transform('count')
    data['sum_area'] = data.groupby('COMID')['area'].transform('sum')
    data['weight'] = data['area']/data['sum_area']
    data.drop('sum_area', axis=1, inplace=True)

    f2 = gpd.read_file('level_01/pfaf_%02d'%i+'_cat_3sMERIT.shp')
    df_id = pd.DataFrame(f2['COMID'],columns=['COMID'])

    final_data = pd.merge(df_id, data, on='COMID', how='inner')
    print(str(len(data))+' reduced to '+str(len(final_data)))
    final_data.fillna(0,inplace=True)
    final_data['ix'] = final_data['ix'].astype(int)
    final_data['iy'] = final_data['iy'].astype(int)
    final_data['number'] = final_data['number'].astype(int)
    final_data = final_data[['COMID','area','ix','iy','number','weight']]
    
    #output file weight table
    final_data.to_csv('tables/weight_table_pfaf_%02d'%i+'_5km.csv',index=False)



