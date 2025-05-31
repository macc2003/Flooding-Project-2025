import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def get_slope(run_title):
    slope = compute_and_plot_slope(f"{run_title}/input/dem_{run_title}.tif")
    output_directory = f"{run_title}/produced" # Ensure this folder exists or create it
    output_filename = f"slope_{run_title}.tif"  # Desired output file name

    # Save the raster
    save_raster(output_directory, output_filename, slope)

def compute_and_plot_slope(input_dem_path):
    with rasterio.open(input_dem_path) as src:
        dem = src.read(1, masked=True)
        transform = src.transform
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # usually negative, convert to positive

        # Compute gradient
        dz_dy, dz_dx = np.gradient(dem, pixel_size_y, pixel_size_x)

        # Compute slope magnitude (in elevation units per meter)
        slope = np.sqrt(dz_dx**2 + dz_dy**2)

    # # Plotting the slope
    # plt.figure(figsize=(10, 6))
    # slope_plot = plt.imshow(slope, cmap='terrain', origin='upper', vmin = 0, vmax = 10)
    # plt.colorbar(slope_plot, label='Slope (elevation change per meter)')
    # plt.title('Slope Magnitude from DEM')
    # plt.xlabel('Pixel Column')
    # plt.ylabel('Pixel Row')
    # plt.tight_layout()
    # plt.show()

    return slope

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

