from function_file import SatelliteData
import matplotlib.pyplot as plt
import numpy as np
import tqdm

"""WE can define the number of pixels of sentinel image, and then the downscale we want!"""

n = 50
seeds = np.arange(1,n)    

sentinel_list = []
lidar_list = []

for seed in tqdm.tqdm(seeds): 
    model = SatelliteData(seed = seed,pixels=128, downscale = 32 ,river_width = 4)
    """THe magnitude is the density of the resulting Lidar image, shift is the lateral shift, blur_factor is how much gaussian noise we wish to add"""
    sentinel, Lidar = model.run(shift = 10, mag = 0.5, blur_factor = 0.2)

    sentinel_list.append(sentinel)
    lidar_list.append(Lidar)

