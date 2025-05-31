import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_drainage_rate(run_title):
    sand_path = f"{run_title}/resampled/sand_rpj_{run_title}_resampled.tif"
    clay_path = f"{run_title}/resampled/clay_rpj_{run_title}_resampled.tif"
    sand = read_raster(sand_path)[0]
    clay = read_raster(clay_path)[0]

    drain_rate_data = get_drainage_rates_data()

    drainage_array = np.zeros((len(sand), len(sand[0])))

    for i in range(len(sand)):
        for j in range(len(sand[0])):
            if sand[i, j] % 2 == 0:
                sand_content = sand[i, j]
            else:
                sand_content = sand[i, j] - 1

            if clay[i, j] % 2 == 0:
                clay_content = clay[i, j]
            else:
                clay_content = clay[i, j] + 1

            drainage_array[i, j] = get_drainage(sand_content, clay_content, drainage_rates=drain_rate_data)

    drainage_array = drainage_array * 10 / (4 * 24 * 60 ** 2)

    # print(drainage_array)
    # plt.imshow(drainage_array)
    # plt.colorbar()
    # plt.show()

    save_raster(f"{run_title}/produced", f"drainage_rate_{run_title}.tif", drainage_array)

def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

def get_drainage_rates_data():
    """ This function opens the csv file specified inside the function and returns the populated dictionary"""

    import csv

    dr_data = {}

    with open('Simple Drainage Rates Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Get data from each row of csv file
            sand_no = int(row["sand"])
            clay_pct = int(row["clay"])
            drain_val= float(row["drainage"])

            # Add the dr_key (sand + clay/100) to the dictionary as the key with the drainage as the value
            dr_key = sand_no + (clay_pct/100)
            dr_data[dr_key] = drain_val

    return dr_data


def get_drainage(sand, clay, drainage_rates):
    """ This function returns the drainage value for the give sand and clay values from the given drainage
    data dictionary which has been previously created with the get_drainage_rates_data function"""

    drainage_key = sand + clay/100
    # Note using drain_rate_data.get to access the data in the dictionary as the get function returns None
    # if the key (sand_clay) doesn't exit in the data
    drainage_value = drainage_rates.get(drainage_key)
    if drainage_value == None:
        drainage_value = 0
    return drainage_value


def save_raster(output_dir, output_filename, array):
    """
    Saves the provided array to a raster file without using the profile.

    Parameters:
    - output_dir: Directory where the file will be saved.
    - output_filename: Name of the output file.
    - array: 2D array of raster data to save.
    """
    # Construct full output path
    output_path = os.path.join(output_dir, output_filename)

    # Check if path is correct
    print(f"Saving to: {output_path}")

    # Define the basic raster metadata manually
    height, width = array.shape  # Shape of the array (height and width)
    transform = rasterio.Affine(1, 0, 0, 0, -1, 0)  # Set a dummy affine transform (adjust if needed)
    crs = 'EPSG:4326'  # Example CRS, change if needed (e.g., 'EPSG:4326' for WGS 84)

    # Save raster to file
    try:
        with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype='float32',
                           width=width, height=height, crs=crs, transform=transform) as dst:
            dst.write(array, 1)  # Write the array to the first band
            print(f"Successfully saved {output_filename}")
    except Exception as e:
        print(f"Error saving file: {e}")
