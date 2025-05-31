import numpy as np
import matplotlib.pyplot as plt
import random
import rasterio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import re
import sys

geometry = [[-77.55, -9.6],
                [-77.375, -9.6],
                [-77.375, -9.45],
                [-77.55, -9.45]]
# slope_adj_CN = []
# equiv_flat_grassland_CN_array = []
# dem = []
# ET_rate = []
# usda_array = []
# wilting_point_array = []
# saturation_point_array = []
# saturated_conductivity = []
# field_capacity_array = []
# manning_n_array = []
# snowmelt_array = []
# drainage_array = []




def run_cpu_md8_slow(run_title, iterations):
    start_time = time.time()
    # Open the TIFF file using Rasterio
    with rasterio.open(f'{run_title}/produced/slope_adjusted_CN_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        slope_adj_CN = src.read(1)  # '1' reads the first band (in case of multi-band TIFFs)

    with rasterio.open(f'{run_title}/produced/grassland_cn_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        equiv_flat_grassland_CN_array = src.read(1)  # '1' reads the first band (in case of multi-band TIFFs)

    with rasterio.open(f'{run_title}/input/dem_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        dem = src.read(1)  # '1' reads the first band (in case of multi-band TIFFs)
    dem = dem * 1000.0
    print(dem)

    with rasterio.open(f'{run_title}/produced/et_water_loss_rate_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        ET_rate = src.read(1)  # Per 30s (so that flow speed is roughly 1m/s)


    with rasterio.open(f'{run_title}/produced/usda_classification_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        usda_array = src.read(1)

    with rasterio.open(f'{run_title}/produced/wilting_point_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        wilting_point_array = src.read(1)

    with rasterio.open(f'{run_title}/produced/saturation_point_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        saturation_point_array = src.read(1)

    with rasterio.open(f'{run_title}/produced/saturated_conductivity_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        saturated_conductivity = src.read(1)

    with rasterio.open(f'{run_title}/resampled/fc_rpj_{run_title}_resampled.tif') as src:
        # Read the image as a NumPy array (first band by default)
        field_capacity_array = src.read(1)

    field_capacity_array = field_capacity_array / 100

    with rasterio.open(f'{run_title}/produced/manning_n_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        manning_n_array = src.read(1)

    with rasterio.open(f'{run_title}/resampled/snowmelt_rpj_{run_title}_resampled.tif') as src:
        # Read the image as a NumPy array (first band by default)
        snowmelt_array = src.read(1)  # Note this is DAILY melt rate

    with rasterio.open(f'{run_title}/produced/drainage_rate_{run_title}.tif') as src:
        # Read the image as a NumPy array (first band by default)
        drainage_array = src.read(1)


    geometry = [[-77.55, -9.6],
                [-77.375, -9.6],
                [-77.375, -9.45],
                [-77.55, -9.45]]

    cn_data = get_cn_data()

    CN_I_array = np.zeros((len(dem), len(dem[0])))
    CN_II_array = np.zeros((len(dem), len(dem[0])))
    CN_III_array = np.zeros((len(dem), len(dem[0])))

    for i in range(len(dem)):
        for j in range(len(dem[0])):
            CN_I_array[i, j] = cn_data[int(slope_adj_CN[i, j])][0]
            CN_II_array[i, j] = cn_data[int(slope_adj_CN[i, j])][1]
            CN_III_array[i, j] = cn_data[int(slope_adj_CN[i, j])][2]

    grassland_infiltration_data = get_infiltration_data()
    grassland_infiltration_data = list(map(lambda x: x * 10 / 60 ** 2, grassland_infiltration_data))

    plt.ion()

    # Define the 8 possible directions for the D8 model (clockwise from north)
    directions = [
        (0, 0),  # Nowhere
        (-1, 0),  # North
        (-1, 1),  # Northeast
        (0, 1),  # East
        (1, 1),  # Southeast
        (1, 0),  # South
        (1, -1),  # Southwest
        (0, -1),  # West
        (-1, -1)  # Northwest
    ]

    size = [559, 646]
    barrier = 10000000
    resolution = 30000

    river_head = np.zeros((len(dem), len(dem[0])))

    for i in range(102, 106):
        for j in range(376, 380):
            river_head[i, j] = 30000

    precipitation = generate_precipitation_grid(size)

    head = generate_head_grid(size)

    water_level_grid = generate_water_level_grid(size)

    accumulated_infiltration = generate_accumulated_infiltration_grid(size)

    I_A = generate_initial_abstraction_grid(size)

    max_speed_array = np.zeros((len(dem), len(dem[0])))

    M = size[0]  # number of rows
    N = size[1]  # number of columns

    # for idx, (di, dj) in enumerate(directions):
    #     ni, nj = i + di, j + dj
    #     flow_speed_array[1][1].append(idx)
    #
    # print(flow_speed_array[1][1])

    T = 0
    dt = 0
    iteration = 0

    # Example usage
    # coord_str = "9°31'29\"S 77°32'10\"W"
    coord_str = "9°31'28\"S 77°32'10\"W"
    # coord_str = "9°27'35\"S 77°25'42\"W"
    lat, long = extract_and_convert_dms(coord_str)
    coords = [coordinate_transformation(lat, long)[0], coordinate_transformation(lat, long)[1]]

    hydrograph = []
    run_model(iterations)

    plt.ioff()
    plt.show()


def read_raster(path):
    with rasterio.open(path) as src:
        array = src.read(1)  # Read first band
        profile = src.profile
    return array, profile

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


# Function for converting AMCII CN to AMC I/III
def get_cn_data():
    # Read data in from a CSV file with headers and put data into a list
    import csv

    cn_list = []
    amc_check = 0
    with open('Curve Numbers Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)

        # Get data from each row of csv file
        for row in reader:
            amc2 = int(row["CN (AMCII)"])

            # Check that the next row of data has increased AMCII number by 1 before getting other data values
            if amc2 == amc_check:
                amc1 = int(row["CN (AMCI)"])
                amc3 = int(row["CN (AMCIII)"])

                # Append the values to the cn_list as a list of 3 valuses (AMCI, AMCII, AMCIII)
                cn_list.append([amc1, amc2, amc3])
                amc_check += 1
            else:
                print("Error in CSV Data - AMC II number out of sequence!")
                return []

        return cn_list

def get_infiltration_data():
    # Read data in from a CSV file with headers and put data into a list
    import csv

    infiltration_data = []
    sti_check = 0

    with open('Infiltration Rates Import.csv', mode ='r') as file:
        reader = csv.DictReader(file)

        # Get data from each row of csv file
        for row in reader:
            soil_type_index = int(row["Soil Type Index"])

            # Check that the next row of data has increased AMCII number by 1 before getting other data values
            if  soil_type_index == sti_check:
                infiltration_data.append(float(row["Grassland (mm/hr)"]))
                sti_check += 1
            else:
                print("Error in CSV Data - Infiltration Data soil type index out of order or missing!")
                return []

    return infiltration_data

def coordinate_transformation(lat, long):

    max_lat = geometry[0][1]
    min_lat = geometry[2][1]
    max_long = geometry[0][0]
    min_long = geometry[1][0]

    if lat<max_lat or lat>min_lat:
        sys.exit("Point outside of simulation area")

    if long < max_long or long > min_long:
        sys.exit("Point outside of simulation area")

    print(f"dem in M_...{dem}")

    x = int(((long-max_long)/abs(max_long-min_long)*len(dem[0]))//1)
    y = int(abs(lat-min_lat)/abs(max_lat-min_lat)*len(dem)//1)

    print(abs(lat-min_lat))
    print(abs(max_lat-min_lat))

    return [x,y]

def generate_head_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    head = 0*x*y

    return head

def generate_river_head_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    mod_xy = (x-0.5)**2 + (y+0.95)**2

    river_head = np.where(mod_xy < 0.0001, 10000, 0)

    return river_head

# river_head = generate_river_head_grid(size)

def generate_precipitation_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    mod_xy = (x)**2 + (y)**2

    ppt = np.where(mod_xy < 2, 10, 0)

    return ppt


def generate_CN_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    CN = 0*x*y + 100

    return CN

# CN_grid = generate_CN_grid(size)

def generate_water_level_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    water_level = 0*x*y

    return water_level

# CN_grid = generate_CN_grid(size)

def generate_accumulated_infiltration_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    accumulated_infiltration = 0*x*y

    return accumulated_infiltration


def generate_initial_abstraction_grid(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size[1]), np.linspace(-1, 1, size[0]))

    initial_abstraction = 0*x*y

    return initial_abstraction


def flow_speed(n, height, slope):
    v = 1/n*height**(2/3)*slope**(1/2)
    return v

def md8_flow_direction(dem):
    # Initialize a matrix to store the flow direction
    flow_direction = np.zeros_like(dem, dtype=int)
    flow_direction_1 = np.zeros_like(dem, dtype=int)

    propagation_array = [[[] for _ in range(N)] for _ in range(M)]
    flow_speed_array = [[[] for _ in range(N)] for _ in range(M)]

    global iteration
    iteration += 1

    global head

    global dt

    global T

    rows, cols = dem.shape
    # print(head)
    # Loop through each cell in the DEM (except the borders)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            wilting_point = wilting_point_array[i,j]
            field_capacity = field_capacity_array[i,j]
            saturation_point = saturation_point_array[i,j]

            if accumulated_infiltration[i,j]< 1000*wilting_point:
                CN = CN_I_array[i,j]

            elif accumulated_infiltration[i,j] > 1000*field_capacity:
                CN = CN_II_array[i,j]

            elif accumulated_infiltration[i,j] > 1000*wilting_point and accumulated_infiltration[i,j]< 1000*field_capacity:
                CN = CN_III_array[i,j]

            if iteration % 3 == 1:
                head[i, j] += precipitation[i, j]


            # head[i,j] += river_head[i,j]

            head[i,j] += snowmelt_array[i,j]*(1000*dt)/(24*60**2) #Conversion from m/day to mm per dt

            # max_retention, S is measured in mm and 25.4 is from conversion from inches to mm
            if CN != 0:
                max_retention = 25400 / CN - 254

            # To avoid /0, may need to tweak this value
            else:
                max_retention = 25400 / 0.0001 - 254


            grassland_CN = equiv_flat_grassland_CN_array[i,j]
            grassland_max_retention = 25400/ grassland_CN - 254

            if accumulated_infiltration[i][j] > saturation_point*1000:
                infiltration = saturated_conductivity[i][j]*dt
                runoff = head[i,j] - infiltration

            else:
                long_time_runoff = head[i,j] ** 2 / (
                            head[i,j] + max_retention)
                long_time_infiltration = head[i,j] - long_time_runoff

                grassland_long_time_runoff = head[i,j] ** 2 / (
                            head[i,j] + grassland_max_retention)
                grassland_long_time_infiltration = head[i,j] - grassland_long_time_runoff

            usda = int(usda_array[i,j])
            infiltration = long_time_infiltration/grassland_long_time_infiltration * grassland_infiltration_data[usda]*dt

            infiltration = min(infiltration, head[i,j])
            runoff = head[i,j] - infiltration
            head[i,j] -= infiltration

            accumulated_infiltration[i,j] += infiltration
            accumulated_infiltration[i,j] -= ET_rate[i,j]*dt

            if accumulated_infiltration[i,j] >= 1000* field_capacity:
                accumulated_infiltration[i,j] -= drainage_array[i,j]*dt
                accumulated_infiltration[i,j] = max(field_capacity, accumulated_infiltration[i,j])

            accumulated_infiltration[i,j] = max(0, accumulated_infiltration[i,j])



            current_elevation = dem[i, j] + head[i, j]

            # List to store neighbours' elevations and their directions
            neighbours = []
            gradient = []

            for idx, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                mod = resolution*(di ** 2 + dj ** 2) ** 0.5

                if mod == 0:
                    grad = 0
                else:
                    total_neighbouring_elevation = dem[ni, nj] + head[ni, nj]
                    grad = (total_neighbouring_elevation - current_elevation) / mod

                gradient.append((grad, idx))
                neighbours.append((dem[ni, nj] + head[ni, nj], idx))

            # print(neighbours)
            # print(gradient)

            total_negative_gradient = 0
            flow_weights = np.empty(9)

            for k in range(9):
                if gradient[k][0] < 0:
                    total_negative_gradient += gradient[k][0]

            flow_speed = 0

            for k in range(9):
                # print(gradient[i][0])


                if gradient[k][0] >= 0:
                    weight = 0
                else:
                    weight = gradient[k][0] / total_negative_gradient

                flow_weights[k] = weight

                ni, nj = i + directions[k][0], j + directions[k][1]

                if gradient[k][0] < 0:
                    n = manning_n_array[i][j]
                    speed = 1/n * (runoff)**(2/3)*np.sqrt(-1*gradient[k][0])

                else:
                    speed = 0

                if speed > flow_speed:
                    flow_speed = speed

                propagation = weight * head[i][j]
                propagation_array[i][j].append(propagation)
                flow_speed_array[i][j].append(speed)

            max_speed_array[i][j] = flow_speed

    max_flow_speed = np.max(max_speed_array)

    dt = resolution/max_flow_speed

    T += dt


    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            for idx, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                flow_factor = flow_speed_array[i][j][idx]/max_flow_speed
                head[ni,nj] += propagation_array[i][j][idx] * flow_factor
                head[i,j] -= propagation_array[i][j][idx] * flow_factor

            if head[i][j] < 0:
                print('NEGATIVE HEAD')
                print(head[i][j])
                print(propagation_array[i][j])

    return

def dms_to_decimal(deg, min, sec, direction):
    decimal = int(deg) + int(min) / 60 + float(sec) / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_and_convert_dms(coord_str):
    # Match DMS pattern like 9°31'29"S
    pattern = r"(\d+)°(\d+)'(\d+)[\"′]([NSWE])"
    matches = re.findall(pattern, coord_str)

    if len(matches) != 2:
        raise ValueError("Expected two DMS coordinates in the string.")

    lat_dms, lon_dms = matches
    lat = dms_to_decimal(*lat_dms)
    lon = dms_to_decimal(*lon_dms)
    return lat, lon

def plot_flow_accumulation(pause,iteration):
    plt.imshow(np.log1p(head), origin = 'upper', cmap = 'Blues')
    plt.colorbar(label = "Surface Water Level")
    print(np.max(accumulated_infiltration))
    # plt.title('Flow Accumulation')
    plt.plot(coords[0], coords[1], 'ro', markersize=8)  # 'ro' = red circle marker
    plt.draw()
    plt.pause(pause)  # Allows real-time update
    plt.close()

def plot_flow_speed(pause,iteration):
    # plt.imshow(np.log1p(head), cmap='Blues', origin='upper')
    plt.imshow(speed_array/1000, origin = 'upper')
    plt.colorbar(label = "Flow speed m/s")
    # plt.colorbar(label='Flow Accumulation')
    # plt.title('Flow Accumulation')
    plt.plot(coords[0], coords[1], 'ro', markersize=8)  # 'ro' = red circle marker
    plt.draw()
    plt.pause(pause)  # Allows real-time update
    plt.close()

def plot_hydrograph(pause,iteration):
    plt.plot(hydrograph)
    if iteration == iterations - 1:
        plt.savefig('Hydrograph.png')
    plt.draw()
    plt.pause(pause)  # Allows real-time update
    plt.close()

def run_model(iterations):
    for i in range(iterations):
        md8_flow_direction(dem)
        hydrograph.append(head[coords[1], coords[0]])
        if i % 1 == 0:
            plot_flow_accumulation(1,i)
            # plot_flow_speed(1,i)
            current_time = time.time()
            runtime = current_time - start_time
            print(f"{i}: {runtime}")
            print(f"T: {T}")
            print(f"dt: {dt}")
        if iteration == iterations - 1:
            save_raster('Medium', 'final head array.tif', head)


