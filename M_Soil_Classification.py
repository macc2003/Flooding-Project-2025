import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio

def get_soil_data(run_title):
    sand_path = f"{run_title}/resampled/sand_rpj_{run_title}_resampled.tif"
    clay_path = f"{run_title}/resampled/clay_rpj_{run_title}_resampled.tif"

    # Load the .tif files
    with rasterio.open(clay_path) as clay_src:
        clay = clay_src.read(1)
        profile = clay_src.profile  # Save this if you want to write output later

    with rasterio.open(sand_path) as sand_src:
        sand = sand_src.read(1)

    # Assume this is your USDA soil class map
    usda = classify_soil_usda_array(sand, clay)

    # Step 1: Define all possible classes
    usda_classes = [
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"]

    # Ensure uniqueness and sort nicely
    usda_classes = sorted(set(usda_classes))

    # Step 2: Map class names to integers
    usda_int = np.full(usda.shape, -1)

    for i, name in enumerate(usda_classes):
        usda_int[usda == name] = i

    # # Step 3: Create a color map
    # n_classes = len(usda_classes)
    # colors = plt.cm.tab20(np.linspace(0, 1, n_classes))  # or try 'tab10', 'Set3', etc.
    # cmap = mcolors.ListedColormap(colors)
    # bounds = np.arange(n_classes + 1) - 0.5
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)
    #
    # # Step 4: Plot
    # plt.figure(figsize=(12, 7))
    # im = plt.imshow(usda_int, cmap=cmap, norm=norm)
    # cbar = plt.colorbar(im, ticks=np.arange(n_classes))
    # cbar.ax.set_yticklabels(usda_classes)
    # plt.title('USDA Soil Texture Classification')
    # plt.axis('off')
    # plt.show()

    wilting_point_array = calculate_wilting_point_array(sand, clay)

    saturation_point_array = calculate_saturation_point_array(sand, clay)

    # Apply the function
    hsg = classify_soil_hsg_array(sand, clay)

    # Step 1: Map group letters to integers
    hsg_classes = ["A", "B", "C", "D", "Unknown"]
    hsg_int = np.full(hsg.shape, -1)  # -1 for Unknown

    for i, group in enumerate(hsg_classes):
        hsg_int[hsg == group] = i

    # # Step 2: Define color map
    # cmap = mcolors.ListedColormap(['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#bdbdbd'])  # you can customize
    # bounds = np.arange(len(hsg_classes) + 1) - 0.5
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # # Step 3: Plot
    # plt.figure(figsize=(10, 6))
    # im = plt.imshow(hsg_int, cmap=cmap, norm=norm)
    # cbar = plt.colorbar(im, ticks=np.arange(len(hsg_classes)))
    # cbar.ax.set_yticklabels(hsg_classes)
    # plt.title('Hydrologic Soil Groups')
    # plt.axis('off')
    # plt.show()

    # Step 4: Convert HSG categories to integers for raster export
    hsg_to_int = {"A": 1, "B": 2, "C": 3, "D": 4, "Unknown": 5}
    hsg_int_export = np.vectorize(hsg_to_int.get)(hsg)

    # Step 5: Update raster profile for writing
    export_profile = profile.copy()
    export_profile.update({
        'dtype': 'uint8',
        'count': 1,
        'compress': 'lzw'
    })

    # Example usage
    output_directory = f"{run_title}/produced"  # Ensure this folder exists or create it
    wp_output_filename = f"wilting_point_{run_title}.tif"  # Desired output file name
    sp_output_filename = f'saturation_point_{run_title}.tif'
    usda_ouput_filename = f'usda_classification_{run_title}.tif'
    hsg_ouput_filename = f'hsg_classification_{run_title}.tif'

    # Save the raster
    save_raster(output_directory, wp_output_filename, wilting_point_array)
    save_raster(output_directory, sp_output_filename, wilting_point_array)
    save_raster(output_directory, usda_ouput_filename, usda_int)
    save_raster(output_directory, hsg_ouput_filename, hsg_int)


# print(classify_soil_usda_single(53,20))
# # print(classify_soil_usda_single(51,18))
def classify_soil_usda_array(sand, clay):
    silt = 100 - sand - clay

    # Initialize with 'Unknown'
    result = np.full(sand.shape, 'Unknown', dtype=object)

    # Each condition from your original function, now vectorized:
    result[(clay >= 40) & (silt >= 40)] = 'Silty Clay'
    result[(clay >= 40) & (sand <= 45)] = 'Clay'
    result[(clay >= 35) & (sand > 45)] = 'Sandy Clay'
    result[(clay >= 27) & (clay < 40) & (sand >= 20) & (sand <= 45)] = 'Clay Loam'
    result[(clay >= 20) & (clay < 35) & (sand > 45) & (silt <= 27)] = 'Sandy Clay Loam'
    result[(clay >= 7) & (clay < 27) & (sand <= 52) & (silt >= 27) & (silt <= 50)] = 'Loam'
    result[(sand >= 85) & (silt + 1.5 * clay < 15)] = 'Sand'
    result[(sand >= 70) & (silt + 1.5 * clay >= 15) & (silt + 2 * clay < 30)] = 'Loamy Sand'
    result[(silt >= 50) & (clay >= 27)] = 'Silty Clay Loam'
    result[(clay >= 27) & (clay < 40) & (sand <= 20)] = 'Loam'
    result[(clay <= 7) & (silt < 50) & (silt + 2 * clay >= 30)] = 'Sandy Loam'
    result[(clay > 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30)] = 'Sandy Loam'
    result[(clay > 14) & (clay < 27) & (silt > 50)] = 'Silt Loam'
    result[(clay <= 14) & (silt > 50) & (silt < 80)] = 'Silt Loam'
    result[(clay <= 14) & (silt >= 80)] = 'Silt'

    return result

def classify_soil_hsg_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)  # should return 2D array of USDA soil classes

    # Initialize output array
    hsg = np.full(usda.shape, 'Unknown', dtype=object)

    # Define USDA groupings
    A = ["Sand"]
    B = ["Loamy Sand", "Sandy Loam"]
    C = ["Sandy Clay Loam", "Clay Loam", "Silty Clay Loam", "Loam", "Silt Loam", "Silt"]
    D = ["Sandy Clay", "Silty Clay", "Clay"]

    # Apply the classifications using np.isin
    hsg[np.isin(usda, A)] = "A"
    hsg[np.isin(usda, B)] = "B"
    hsg[np.isin(usda, C)] = "C"
    hsg[np.isin(usda, D)] = "D"

    return hsg

def calculate_wilting_point_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)  # 2D array of USDA classes

    usda_classes = np.array([
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"
    ])

    wilting_points = np.array([
        0.04, 0.06, 0.08, 0.13, 0.25, 0.22, 0.10, 0.13, 0.13,
        0.13, 0.27, 0.28, 0.14  # default or placeholder for 'Unknown'
    ])

    # Initialize output array
    wilting_point_array = np.full(usda.shape, 0.0)

    for i, soil_class in enumerate(usda_classes):
        wilting_point_array[usda == soil_class] = wilting_points[i]

    return wilting_point_array

def calculate_saturation_point_array(sand, clay):
    usda = classify_soil_usda_array(sand, clay)  # 2D array of USDA classes

    usda_classes = np.array([
        "Sand", "Loamy Sand", "Sandy Loam", "Sandy Clay Loam", "Clay Loam",
        "Silty Clay Loam", "Loam", "Silt Loam", "Silt", "Sandy Clay",
        "Silty Clay", "Clay", "Unknown"
    ])

    saturation_points = np.array([
       0.375, 0.39 , 0.387, 0.384, 0.442, 0.482, 0.399, 0.439, 0.489,
       0.385, 0.481, 0.459, 0.426  # Average for unknown
    ])

    # Initialize output array
    saturation_point_array = np.full(usda.shape, 0.0)

    for i, soil_class in enumerate(usda_classes):
       saturation_point_array[usda == soil_class] = saturation_points[i]

    return saturation_point_array


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


