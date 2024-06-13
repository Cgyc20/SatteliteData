from function_file import SatelliteData
import matplotlib.pyplot as plt
import numpy as np
import tqdm

"""WE can define the number of pixels of sentinel image, and then the downscale we want!"""

seed = 5
model = SatelliteData(seed = seed,pixels=128, downscale = 32 ,river_width = 4)
"""THe magnitude is the density of the resulting Lidar image, shift is the lateral shift, blur_factor is how much gaussian noise we wish to add"""
sentinel, Lidar = model.run(shift = 10, mag = 0.5, blur_factor = 0.2)

_, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(sentinel, cmap='Greens')
axs[0].set_title("Sentinel")

axs[1].imshow(Lidar, cmap='Greens')
axs[1].set_title("LIDAR")
axs[1].axis('off')

plt.show()