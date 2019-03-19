# GlobalHydro
Global high-resolution streamflow modeling at 2.94 million river reaches extracted from ~90 m DEM

1. Set up global routing model configurations and prepare inputs for the RAPID model:
(1) make_intersect.py #intersect unit catchment with LSM grids
(2) make_weight.py #generate weight table
(3) fast_connectivity.py #extract topology (river connectivity) using flowline shapefiles

2. Bias correction against nine runoff charactersistics maps derived from machine learning
(1) generate_pixels.py #extract all lat/lon info for grid cells that need BC (in this case ~0.24 million VIC grid cells)
(2) main_BC.py #main program for bias correction (see Lin et al. 2019; WRR submitted) for more details
(3) control.py #control jobs/cores to divide the entire domain (in this example, 40 jobs 10 cores are used for faster processing)

3. Generate property tables for shapefile
(1) calc_order.py #extract Strahler stream order for flowlines based on calculate connectivity table (see #1)
(2) calc_area.py #calculate the unit catchment area in km2 using AEA (Albert Equal Area) projection

More info: contact Peirong Lin (peirongl@princeton.edu) for questions
