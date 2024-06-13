import numpy as np
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import random
import scipy.ndimage
import cv2

class SatelliteData:
    def __init__(self, seed, pixels, downscale,river_width):
        self.seed = seed 
        self.pixels = pixels
        self.downscale = downscale
        self.river_width = river_width
        random.seed(seed)
        
        self.noise1 = PerlinNoise(octaves=3, seed=seed)
        self.noise2 = PerlinNoise(octaves=16, seed=seed)
        self.sentinel = self._generate_sentinel()

    def _generate_sentinel(self):
        """Generates the sentinel data of certain pixel size"""
        sentinel = np.zeros((self.pixels, self.pixels))
        for i in range(self.pixels):
            for j in range(self.pixels):
                noise_val = self.noise1([i / self.pixels, j / self.pixels])
                noise_val += 0.2 * self.noise2([i / self.pixels, j / self.pixels])
                sentinel[i, j] = noise_val

        def river_poly(x, river_seed):
            """The river creating polynomial - Sinusoidal with random amplitude and phase shift"""
            random.seed(river_seed)
            amplitude_factor = random.uniform(0.5, 1.5)  # Random factor for amplitude variation
            phase_shift = random.uniform(-0.5, 0.5)  # Random phase shift
            frequency_factor = random.uniform(0.8, 1.2)  # Optional: Random factor for frequency variation

            return int(self.pixels // 2 + (self.pixels / 15) * amplitude_factor * (1 + np.sin(2 * np.pi * frequency_factor * (x + phase_shift) / self.pixels) + random.uniform(-1, 1)))

        # Apply the river path
        river_seed = self.seed + 1
        for i in range(self.pixels):
            y = river_poly(i,river_seed)
            sentinel[i, max(0, y-self.river_width):min(self.pixels, y+self.river_width)] = -0.8  # Making the river a bit wider and setting to -0.8 to distinguish

        # Normalize the noise values to range between 0 and 1
        sentinel = (sentinel - np.min(sentinel)) / (np.max(sentinel) - np.min(sentinel))
        # Add a little gaussian blur
        sentinel = scipy.ndimage.gaussian_filter(sentinel, 0.1)
        
        return sentinel

    def _generate_lidar(self, shift, mag, blur_factor):
        """This will adapt the lidar from the original_image (sentinel) 
        The shift gives the lateral shift of the image
        the magnitude is the factor that we want to multiply the lidar by (smaller means lower density) 
        And finally the blur factor will give the amount of gaussian noise we want to add"""
        lidar = self.sentinel.copy()
        lidar = scipy.ndimage.gaussian_filter(lidar, blur_factor)
        lidar = np.roll(lidar, shift, axis=1)
        lidar *= mag

        x = np.linspace(0, 1, self.pixels)
        y = np.linspace(0, 1, self.pixels)
        X, Y = np.meshgrid(x, y)

        u1 = 0.05 * np.sin(4 * np.pi * X) #Add the deformation 
        u2 = np.zeros_like(u1)

        lidar = self._deform_np(lidar, X, Y, u1, u2)

        res = cv2.resize(lidar, dsize=(self.downscale, self.downscale), interpolation=cv2.INTER_CUBIC)
        lidar_resized = cv2.resize(res, dsize=(self.pixels, self.pixels), interpolation=cv2.INTER_CUBIC)

        return lidar_resized

    @staticmethod
    def _deform_np(im1, X, Y, ux, uy):

        """This adds some small deformation to the image"""
        width = im1.shape[1]
        height = im1.shape[0]
        
        # Coordinates to pixel. Assumes domain [0,1]x[0,1]
        X_new = (X + ux) * (width - 1)
        Y_new = (Y + uy) * (height - 1)

        X0 = np.floor(X_new)
        X1 = X0 + 1
        Y0 = np.floor(Y_new)
        Y1 = Y0 + 1

        AA = ((X1 - X_new) * (Y1 - Y_new)).ravel()  # 00
        AB = ((X_new - X0) * (Y1 - Y_new)).ravel()  # 10
        AC = ((X_new - X0) * (Y_new - Y0)).ravel()  # 11
        AD = ((X1 - X_new) * (Y_new - Y0)).ravel()  # 01

        X0 = np.clip(X0, 0, width - 1)
        X1 = np.clip(X1, 0, width - 1)
        Y0 = np.clip(Y0, 0, height - 1)
        Y1 = np.clip(Y1, 0, height - 1)

        X0 = X0.astype(int)
        X1 = X1.astype(int)
        Y0 = Y0.astype(int)
        Y1 = Y1.astype(int)

        i00 = X0.ravel() + width * Y0.ravel()
        i10 = X1.ravel() + width * Y0.ravel()
        i11 = X1.ravel() + width * Y1.ravel()
        i01 = X0.ravel() + width * Y1.ravel()

        im1r = im1.ravel()

        return (AA * im1r[i00] + AB * im1r[i10] + AC * im1r[i11] + AD * im1r[i01]).reshape(height, width)
    

    def run(self, shift, mag, blur_factor):

        """This runs the model and will output sentinel and Lidar"""

        lidar = self._generate_lidar(shift, mag, blur_factor)

        return self.sentinel, lidar


