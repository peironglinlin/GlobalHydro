#Peirong Lin (upload 2019-Jan-15)
#Inputs: catchment shapefile for each continent
#Outputs: catchment area in km2
#Prerequisite: geopandas, shapely

import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point, Polygon
import shapely.ops as ops
import pyproj
from functools import partial

for i in range(1,9):
    print(i)
    print('... read catchmentgis.shp file ... wait ...')
    l = gpd.read_file('level_01/pfaf_%02d'%i+'_cat_3sMERIT.shp')
    l.crs = {'init': 'epsg:4326'}
    l['area']=l['geometry'].apply(lambda x: ops.transform(partial(pyproj.transform,pyproj.Proj(init='EPSG:4326'),pyproj.Proj(proj='aea',lat1=x.bounds[1],lat2=x.bounds[3])),x).area/10**6)
    data = l[['COMID','area']]
    print(' writing to weight table ... wait ...')
    data.to_csv('tables/area_catchment_pfaf_%02d'%i+'.csv',index=False)

