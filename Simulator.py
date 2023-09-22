import os
# change directory
os.chdir("C:\\Users\\kniroomand\\Desktop\\BSRGAN")

import ctypes
from ctypes import CDLL, POINTER, c_uint32, c_uint8, c_float
import numpy as np
import time
import subprocess
import torch
import png
from tqdm import tqdm  
import os.path
import logging
import torch
import os
import numpy as np
os.chdir("C:\\Users\\kniroomand\\Desktop\\BSRGAN")
from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
"""

# Define the model and device here (outside of the function)
model = None  # Define your model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and set it up (you should replace this with your model loading code)
def setup_model():
    global model
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)  # Define your model
    model.load_state_dict(torch.load('model_zoo/BSRGAN.pth'), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()

# Call this function to set up the model before using upscale_hogel
setup_model()

# Upscale function (assuming old_hogel has 33 channels)
def upscale_hogel(old_hogel):
    # Convert NumPy array to PyTorch tensor
    img_L = torch.from_numpy(old_hogel).unsqueeze(0).float() / 255.0
    img_L = img_L.permute(0, 3, 1, 2).to(device)  # Adjust channel order and shape if needed

    # Perform inference
    img_E = model(img_L)

    # Convert the output PyTorch tensor to NumPy array
    sr_image = img_E.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
    sr_image = sr_image.clip(0, 255).astype(np.uint8)

    return sr_image


# File directory (if needed)
# file_directory = os.path.dirname(os.path.abspath(__file__))

HDS_lib = CDLL("C:\\Users\\kniroomand\\Desktop\\Codec\HDS\\build\\Debug\\HDS.dll")

start_light_field_load = time.perf_counter()
lightfield = np.fromfile("C:\\Users\\kniroomand\\Downloads\\TextureMonkey33\\TextureMonkey33.rgb888", dtype=np.uint8)
print(f"Time to load .rgb888 light field: {time.perf_counter() - start_light_field_load}")

# Define spatial dimensions and directional sizes
spatial_x = 121
spatial_y = 132
directional = 33
big_directional = 2 * directional

# Initialize y_hogel and x_hogel to zero
y_hogel = 0
x_hogel = 0

# Preallocate memory for arrays on the GPU
old_hogel_cpu = np.zeros((directional, directional, 3), dtype=np.uint8)
big_lightfield_cpu = np.zeros((spatial_y * big_directional, spatial_x * big_directional, 3), dtype=np.uint8)

# Threshold for black background detection (set to 0 for each color channel)
background_threshold = [0, 0, 0]


# Capture the end time for upscaling
start_upscale = time.perf_counter()

# Loop through the lightfield and upscale it with a progress bar
for y_hogel in tqdm(range(spatial_y), desc="Upscaling"):
    for x_hogel in tqdm(range(spatial_x), desc="Columns", leave=False):
        for y in range(directional):
            for x in range(directional):
                for c in range(3):
                    
                    old_hogel_cpu[y, x, c] = lightfield[(y_hogel * directional * spatial_x * 3 * directional) + (x_hogel * directional * 3) + (y * 3 * directional * spatial_x) + (x * 3) + c]
                    
        # Check if the hogel contains foreground information (not pure black)
        if not np.all(old_hogel_cpu <= background_threshold):
            # Upscale the hogel using your upscale function
            big_hogel_cpu = upscale_hogel(old_hogel_cpu)
            big_hogel_cpu_saving = big_hogel_cpu.reshape(132, -1)
            png.from_array(big_hogel_cpu_saving, mode='RGB').save(f'big_hogel_YHogel{y_hogel}_XHogel{x_hogel}.png')

            # Directly perform the operation on the GPU
            for y in range(big_directional):
                for x in range(big_directional):
                    for c in range(3):
                        big_lightfield_cpu[y_hogel * big_directional + y, x_hogel * big_directional + x, c] = big_hogel_cpu[y, x, c]


# Capture the end time for upscaling
end_upscale = time.perf_counter()

# Calculate the execution time for upscaling
execution_time_upscale = end_upscale - start_upscale
print(f"Time for upscaling: {execution_time_upscale} seconds")

# Define parameters for observer_image
lens_pitch = 0.001
viewing_angle_in_radians = 1.2915
observer_resolution_x = 1023  #Dividble by 3
observer_resolution_y = 1023  #Dividble by 3
observer_x = 0.0
observer_y = 0.0
observer_z = 0.4
observer_field_of_view = 1.570795

# Create arrays to store the observer image with the correct dimensions for RGB
observer_image_np = np.zeros((observer_resolution_y, observer_resolution_x, 3), dtype=np.uint8)


hds_count = 100

start_hds_to_png = time.perf_counter()
for i in range(hds_count):
    observer_x = -0.2 + 0.4 * (i / hds_count)

    HDS_lib.HDS(c_uint32(spatial_x),
                c_uint32(spatial_y),
                c_uint32(directional),
                c_float(lens_pitch),
                c_float(viewing_angle_in_radians),
                c_uint32(observer_resolution_x),
                c_uint32(observer_resolution_y),
                c_float(observer_x),
                c_float(observer_y),
                c_float(observer_z),
                c_float(observer_field_of_view),
                big_lightfield_cpu.ctypes.data_as(POINTER(c_uint8)),
                observer_image_np.ctypes.data_as(POINTER(c_uint8))
            )
    
    # Save the observer_image directly from the GPU to a file
    png.from_array(observer_image_np, mode='RGB').save('hds_{:04d}.png'.format(i))

print(f"Time to write {hds_count} png images: {time.perf_counter() - start_hds_to_png}")

start_ffmpeg_video = time.perf_counter()
subprocess.call("ffmpeg -framerate 30 -y -i hds_%04d.png -c:v libx264 -pix_fmt yuv420p hds_video.mp4", stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
print(f"Time to make mp4 video: {time.perf_counter() - start_ffmpeg_video}")
