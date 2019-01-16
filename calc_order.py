#Peirong Lin (upload 2019-Jan-15)
#Inputs: river network connectivity organized as RAPID model input format
#Outputs: stream order

import pandas as pd
import numpy as np

#to avoid maximum recursive limit
import sys
sys.setrecursionlimit(10000)

def calc_order(ID):
    if not ID in df_dict: #only calculate area if it has not been calculated
        if df1_dict[ID][1]==0: #if no maxup, order == 1
            df_dict[ID] = 1
            return df_dict[ID]
        else:
            orderup = []
            for i in range(df1_dict[ID][1]):
                orderup.append(calc_order(df1_dict[ID][i+2]))
            if np.mean(orderup) == orderup[0]: #all orders are the same
                df_dict[ID] = orderup[0]+1
            else:
                df_dict[ID] = int(np.max(orderup))
            return df_dict[ID]
    else:
        return df_dict[ID]
    
for i in range(1,9):
    print(i)
    #read connectivity file
    df1 = pd.read_csv('tables/fast_connectivity_%02d'%i+'.csv',header=None)
    df1.columns=['COMID','NextID','maxup','up1','up2','up3','up4']  
    print(len(df1))
    df1_dict = dict(zip(df1.COMID,list(zip(df1.NextID,df1.maxup,df1.up1,df1.up2,df1.up3,df1.up4))))
    df_dict = {} # used to save calculated IDs
     
    #calculate area for each GRIDCODE
    df1['COMID'].apply(lambda x: calc_order(x))
    
    data = pd.DataFrame({'COMID':list(df_dict.keys()), 'order':list(df_dict.values())})
    data.to_csv('tables/stream_order_%02d'%i+'.csv',index=False)
    
