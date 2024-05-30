######################################################################
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import re
import csv
from keras import backend as K
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from pickle import load
import math
from matplotlib.tri import Triangulation
from scipy.signal import savgol_filter
import mplcursors
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
import pyvista as pv
######################################################################
rcParams['font.size'] = 15
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

################################Global variables #####################
#3600000p
number_of_points = 5400000
radius = 10  # cm
sampling_time = 0.01  # rpt sampling time, millisecond
r_min = 0  # cm
z_min_tank = -20  # cm
z_max_tank = 0  # cm
nr = 70  # in r direction
nz = 140  # in z direction
min_occurrence = 350
density = 1270
# Savitzky-Golay filter parameters
window_size = 31
order = 9
######################################################################


def predict_position():
    """
    Parameters:
    
    -number_of_points: Number of positions
    
    """
    # import all the counts from the trajectory
    file_path = 'counts_all.txt'
    line_count = sum(1 for _ in open(file_path, 'r'))
    print(line_count)
    # Feed file content
    Feed = np.zeros([number_of_points, 9])

    # File for the number of counts
    filename_counts = 'counts_all.txt'
    data_counts = np.loadtxt(filename_counts, delimiter='\t')

    for i in range(9):
        Feed[:, i] = data_counts[0:number_of_points, i]

    filtered_data_counts = np.copy(data_counts)
    for i in range(data_counts.shape[1]):
        # Apply the filter to the i-th column
        filtered_data_counts[:, i] = savgol_filter(data_counts[:, i], 31, 2)

    # load the trained model
    model = load(open('model.pkl', 'rb'))
    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    Pred = filtered_data_counts[-number_of_points:-1, :9]
    X_pre = Pred
    scaled_X_pre = scaler.transform(X_pre)
    (count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre, count7_pre,
     count8_pre, count9_pre) = np.transpose(scaled_X_pre)
    prediction = model.predict([count1_pre, count2_pre, count3_pre, count4_pre, count5_pre, count6_pre,
                                count7_pre, count8_pre, count9_pre])

    # Convert the data to a NumPy array
    predicted_x_pos = np.zeros(number_of_points)
    predicted_y_pos = np.zeros(number_of_points)
    predicted_z_pos = np.zeros(number_of_points)

    for i in range(len(predicted_x_pos) - 1):
        predicted_x_pos[i] = prediction[0][i][0] * 100  # convert to cm
        predicted_y_pos[i] = prediction[1][i][0] * 100
        predicted_z_pos[i] = prediction[2][i][0] * 100

    #np.savetxt('predicted_z_pos.txt', predicted_z_pos, fmt='%.2f', header='Predicted_Z_Pos', comments='')

    # Apply Savitzky-Golay filter

    # Polynomial order
    filtered_x = savgol_filter(predicted_x_pos, window_size, order)
    filtered_y = savgol_filter(predicted_y_pos, window_size, order)
    filtered_z = savgol_filter(predicted_z_pos, window_size, order)

    return predicted_x_pos, predicted_y_pos, predicted_z_pos


def translation():
    """
    to find the origin
    this part first find the origin of the available data and then shift it
    here we also shift x to have the center of x also 0
    then the new origin would be (0,0) instead of (10,0)ish
    """

    """
    Parameters:
    
    -number_of_points: Number of positions
    
    -radius: radius of tank (cm)
    """

    x_translation = radius - np.average(x_pred)
    y_translation = np.average(y_pred)

    translated_x_pos = x_pred + x_translation - radius

    if y_translation < 0:
        translated_y_pos = y_pred - y_translation
    else:
        translated_y_pos = y_pred + y_translation

    #combined_positions = np.column_stack((translated_x_pos, translated_y_pos))

    # Save the combined positions to a text file
    #np.savetxt('translated_positions.txt', combined_positions, fmt='%.2f', delimiter='\t',
               #header='Translated_X_Pos\tTranslated_Y_Pos', comments='')

    return translated_x_pos, translated_y_pos


def xy_to_rz():
    """
    calculate the r and z of each point in trajectory
    """

    x_translated, y_translated = translation()

    r_component = np.zeros(number_of_points)
    theta_component = np.zeros(number_of_points)
    z_component = np.zeros(number_of_points)

    for i in range(0, number_of_points):
        r_component[i] = np.sqrt(np.square(x_translated[i]) + np.square(y_translated[i]))
        theta_component[i] = np.arctan2(y_translated[i], x_translated[i])
        z_component[i] = z_pred[i]

    max_r = np.ceil(r_component.max())
    min_z = -np.ceil(-z_component.min())  # to round up this negative band to a higher negative number
    max_z = np.ceil(z_component.max())

    return r_component, z_component, theta_component, max_r, min_z, max_z


def calculate_the_midpoints():
    x_translated, y_translated = translation()

    x_midpoint = np.zeros(number_of_points - 1)
    y_midpoint = np.zeros(number_of_points - 1)
    z_midpoint = np.zeros(number_of_points - 1)

    for i in range(0, number_of_points - 1):
        x_midpoint[i] = (x_translated[i + 1] + x_translated[i]) / 2
        y_midpoint[i] = (y_translated[i + 1] + y_translated[i]) / 2
        z_midpoint[i] = (z_pred[i + 1] + z_pred[i]) / 2

    r_midpoint = np.zeros(number_of_points - 1)
    theta_midpoint = np.zeros(number_of_points - 1)
    for i in range(0, number_of_points - 1):
        r_midpoint[i] = np.sqrt(np.square(x_midpoint[i]) + np.square(y_midpoint[i]))
        theta_midpoint[i] = np.arctan2(y_midpoint[i], x_midpoint[i])

    return r_midpoint, theta_midpoint, z_midpoint


def midpoints_velocity():
    # calculate the v_r and v_z of midpoints
    r_midpoint, theta_midpoint, z_midpoint = calculate_the_midpoints()

    r, z, theta, r_max, z_min, z_max = xy_to_rz()

    v_r_midpoint = np.zeros(number_of_points - 1)
    v_theta_midpoint = np.zeros(number_of_points - 1)
    v_z_midpoint = np.zeros(number_of_points - 1)

    #calculate deltatheta
    delta_theta = np.zeros(number_of_points - 1)
    for i in range (0,number_of_points-1):
        delta_theta[i] = theta[i + 1] - theta[i]
        if ((theta[i + 1] - theta[i])>np.pi):
            delta_theta[i]= (theta[i + 1] - theta[i])- (2. * np.pi)
        elif ((theta[i + 1] - theta[i])<(-1*np.pi)):
            delta_theta[i] = (theta[i + 1] - theta[i])+ (2. * np.pi)

    for i in range(0, number_of_points - 1):
        v_r_midpoint[i] = (r[i + 1] - r[i]) / sampling_time
        #v_theta_midpoint[i] = r_midpoint[i]*(((theta[i + 1] - theta[i] + np.pi) % (2. * np.pi) - np.pi) / sampling_time)
        v_theta_midpoint[i] = (r_midpoint[i]*delta_theta[i]) / sampling_time
        v_z_midpoint[i] = (z[i + 1] - z[i]) / sampling_time

    return v_r_midpoint, v_theta_midpoint, v_z_midpoint


def mesh():
    r, z, theta, r_max, z_min, z_max = xy_to_rz()

    r_grid = np.linspace(0, r_max, nr)
    z_grid = np.linspace(z_min, z_max, nz)
    rr_grid, zz_grid = np.meshgrid(r_grid, z_grid)

    return rr_grid, zz_grid


def find_cell_index(meshgrid, point):
    x_values = meshgrid[0][0, :]
    y_values = meshgrid[1][:, 0]
    found_x_index = np.searchsorted(x_values, point[0], side='right')-1
    found_y_index = np.searchsorted(y_values, point[1], side='right')-1

    return found_x_index, found_y_index


def assign_velocity_to_cells():

    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    dict_v_r_component = {}
    dict_v_theta_component = {}
    dict_v_z_component = {}
    clean_axial_velocity()
    dict_event = {}

    # initialization
    for i in key_cell_index:
        dict_v_r_component[i] = []

    for i in key_cell_index:
        dict_v_z_component[i] = []

    for i in key_cell_index:
        dict_v_theta_component[i] = []

    for i in range(0, (number_of_points - 1)):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)  # key output the x_index and y_index
            dict_v_r_component[key].append(v_r_mid[i] / 100)
            dict_v_theta_component[key].append(v_theta_mid[i] / 100)
            dict_v_z_component[key].append(v_z_mid[i] / 100)

            #print(f"Key: {key}, v_z_mid[i]: {v_z_mid[i]/100}")
    # Dictionary to store the count of consecutive repetitions for each key
    consecutive_key_count = {}

    previous_key = None
    counter = 1

    for i in range(number_of_points-1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)
            #print(key)
            if key == previous_key:
                counter += 1
            else:
                if previous_key is not None:
                    if previous_key in consecutive_key_count:
                        consecutive_key_count[previous_key].append(counter)
                    else:
                        consecutive_key_count[previous_key] = [counter]
                counter = 1  # Reset counter for the new key
            previous_key = key


    return consecutive_key_count, dict_v_r_component, dict_v_theta_component, dict_v_z_component


def data_for_histogram_axial_velocity():

    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    dict_v_z_component = {}
    for i in key_cell_index:
        dict_v_z_component[i] = []

    for i in range(number_of_points-1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)

    consecutive_key_count = {}
    max_event = []
    min_event = []
    average_event = []
    max_event_values = []

    short_memory_max = []
    short_memory_min = []
    previous_key = None
    counter = 1
    #dict_v_z_component[key].append(v_z_mid[i] / 100)

    for i in range(number_of_points - 1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # To remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)

            if key == previous_key:
                short_memory_max.append(v_z_mid[i] / 100)
                short_memory_min.append(v_z_mid[i] / 100)
                counter += 1
            else:
                if previous_key is not None:
                    if previous_key in consecutive_key_count:
                        consecutive_key_count[previous_key].append(counter)
                    else:
                        consecutive_key_count[previous_key] = [counter]

                    # Print and clear short_memory_max for the previous key
                    #print(previous_key)
                    #print(f"{short_memory_max}")
                    #print(f" {np.max(short_memory_max)}")
                    max_event.append(np.max(short_memory_max))
                    min_event.append(np.min(short_memory_min))
                    short_memory_max.clear()
                    short_memory_min.clear()

                # Initialize short_memory_max and counter for the new key
                short_memory_max.append(v_z_mid[i] / 100)
                short_memory_min.append(v_z_mid[i] / 100)
                counter = 1

            previous_key = key

    plt.hist(max_event, bins='auto', alpha=0.5, edgecolor='black', color='blue', label='Max Event')

    # Create the histogram for min_event
    plt.hist(min_event, bins='auto', alpha=0.5, edgecolor='black', color='red', label='Min Event')

    # Add titles and labels
    plt.title('Histogram of Max and Min Events')
    plt.xlabel('Velocity')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()


def data_for_scatter_plot():

    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    dict_v_z_component = {}
    for i in key_cell_index:
        dict_v_z_component[i] = []

    for i in range(number_of_points-1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # to remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)

    consecutive_key_count = {}
    max_event = []
    min_event = []
    average_event = []

    short_memory_max = []
    short_memory_min = []

    eventlength_velocity = {}
    previous_key = None
    counter = 1
    #dict_v_z_component[key].append(v_z_mid[i] / 100)

    for i in range(number_of_points - 1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # To remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)

            if key == previous_key:
                short_memory_max.append((v_z_mid[i] / 100))
                short_memory_min.append((v_z_mid[i] / 100))
                counter += 1
            else:
                if previous_key is not None:
                    if previous_key in consecutive_key_count:
                        consecutive_key_count[previous_key].append(counter)

                    else:
                        consecutive_key_count[previous_key] = [counter]

                    # Print and clear short_memory_max for the previous key
                    #print(previous_key)
                    #print(f"{short_memory_max}")
                    #print(f" {np.max(short_memory_max)}")

                    if counter in eventlength_velocity:
                        eventlength_velocity[counter].append(np.average(short_memory_max))

                    else:
                        eventlength_velocity[counter] = [np.average(short_memory_max)]

                    max_event.append(np.max(short_memory_max))
                    min_event.append(np.min(short_memory_min))
                    short_memory_max.clear()
                    short_memory_min.clear()

                # Initialize short_memory_max and counter for the new key
                short_memory_max.append((v_z_mid[i] / 100))
                short_memory_min.append((v_z_mid[i] / 100))
                counter = 1

            previous_key = key
    #print(eventlength_velocity)
    # Prepare lists to store the keys and their corresponding values

    for key, values in eventlength_velocity.items():
        print(f'Key: {key}, Length of vector: {len(values)}')

    x = []
    y = []

    # Iterate through the dictionary and collect keys and values
    for key, values in eventlength_velocity.items():
        for value in values:
            x.append(key)
            y.append(value)

    # Create the scatter plot
    plt.scatter(x, y, color='blue', label='Velocity', s=1)

    # Add titles and labels
    plt.title('Event Length vs Velocity')
    plt.xlabel('Event Length')
    plt.ylabel('Velocity')

    # Add a legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()


def calculate_standard_deviation_of_velocity():

    # calculate the SD of velocity in each cell
    event, dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    # v_r
    standard_deviation_v_r = {key: np.std(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_sd = np.array(list(standard_deviation_v_r.values()))  # to convert cm/sec to m/sec
    # v_z
    standard_deviation_v_z = {key: np.std(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_sd = np.array(list(standard_deviation_v_z.values()))

    return standard_deviation_v_r, standard_deviation_v_z, Axial_velocity_sd, Radial_velocity_sd


def average_velocity_at_each_cell():

    event, dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    # average the v_r at each cell
    average_v_r_component = {key: np.mean(values) if values else 0 for key, values in dict_v_r.items()}
    Radial_velocity_component = np.array(list(average_v_r_component.values()))
    # average the v_theta at each cell
    average_v_theta_component = {key: np.mean(values) if values else 0 for key, values in dict_v_theta.items()}
    Tangential_velocity_component = np.array(list(average_v_theta_component.values()))
    #print(len(dict_v_theta))
    # average the v_z at each cell
    average_v_z_component = {key: np.mean(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_component = np.array(list(average_v_z_component.values()))
    max_value = np.max(Axial_velocity_component)

    return average_v_r_component, Radial_velocity_component, average_v_theta_component, Tangential_velocity_component, average_v_z_component, Axial_velocity_component


def find_cell_center(meshgrid, cell_indices):

    x_values = meshgrid[0][0, :]
    z_values = meshgrid[1][:, 0]

    x_center = 0.5 * (x_values[cell_indices[0]] + x_values[cell_indices[0] + 1])
    z_center = 0.5 * (z_values[cell_indices[1]] + z_values[cell_indices[1] + 1])

    return x_center, z_center


def write_cell_center():
    rr, zz = mesh()
    center_point = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            center_point.append(find_cell_center((rr, zz), (i, j)))

    #print(rr,zz)

    return center_point


def number_of_occurrence_at_each_cell():

    event, dict_v_r, dict_v_theta, dict_v_z = assign_velocity_to_cells()
    distribution_dict = {}
    for key, vector in dict_v_r.items():
        # Count the total number of objects in the vector
        total_count = len(vector)
        # Store the total count in the new dictionary
        distribution_dict[key] = total_count

    num_occurrence = list(distribution_dict.values())

    #print(num_occurrence)

    return num_occurrence


def number_of_event_at_each_cell():

    event, dict_v_r_component, dict_v_theta_component, dict_v_z_component = assign_velocity_to_cells()
    distribution_dict = {}
    for key, vector in event.items():
        # Count the total number of objects in the vector
        total_count = len(vector)
        # Store the total count in the new dictionary
        distribution_dict[key] = total_count

    #print(distribution_dict)
    max_key = max(distribution_dict, key=distribution_dict.get)

    # Get the maximum value
    max_value = distribution_dict[max_key]
    #print(max_value)

    return distribution_dict


def plot_occurrence_distribution():

    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):

        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, num_occurrence, cmap="coolwarm", shading='gouraud')
    cbar = plt.colorbar()
    #cbar.set_label('Values')
    # Customize x-axis ticks
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('occurrence_distribution.png', bbox_inches='tight')  # Save the figure
    plt.clf()

    return num_occurrence, center_x, center_y


def plot_event_distribution():

    center_point = write_cell_center()
    num_event = number_of_event_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    new_dict = {}

    # Loop over key_cell_index and append the values to the new dictionary
    for key in key_cell_index:
        if key in num_event:
            new_dict[key] = num_event[key]
        else:
            new_dict[key] = 0

    num_occurrence = list(new_dict.values())

    for i in range(0, (len(center_point))):

        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, num_occurrence, cmap="coolwarm", shading='gouraud', vmin=0, vmax=3500)
    cbar = plt.colorbar()
    #cbar.set_label('Values')
    # Customize x-axis ticks
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('occurrence_distribution.png', bbox_inches='tight')  # Save the figure
    #plt.show()
    plt.clf()

    return num_occurrence


def plot_difference_between_event_occurrence():

    occurrence, center_x, center_y = plot_occurrence_distribution()
    event = plot_event_distribution()
    difference = occurrence - event

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, difference, cmap="coolwarm", shading='gouraud', vmin=200)
    cbar = plt.colorbar()
    #cbar.set_label('Values')
    # Customize x-axis ticks
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('r (cm)')
    plt.ylabel('z (cm)')
    plt.savefig('occurrence_distribution.png', bbox_inches='tight')  # Save the figure
    plt.show()
    plt.clf()


def plot_velocity():

    center_point = write_cell_center()
    num_occurrence = number_of_occurrence_at_each_cell()
    average_v_r_component, Radial_velocity, average_v_theta_component, Tangential_velocity, average_v_z_component, Axial_velocity = average_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        Tangential_velocity = np.delete(Tangential_velocity, reversed_vector[i])
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity[i] = 0
            Tangential_velocity[i] = 0
            Radial_velocity[i] = 0
    max_value = np.max(Axial_velocity)
    #print(max_value)
    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, Axial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')

    plt.ylabel('z (cm)')

    plt.savefig('axial_velocity_contour.png', dpi=500, bbox_inches='tight')  # Save the figure
    plt.show()

    plt.clf()

    plt.tripcolor(triang, Radial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec

    cbar = plt.colorbar()

    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')

    # Set labels and title

    plt.xlabel('r (cm)')

    plt.ylabel('z (cm)')

    plt.savefig('radial_velocity_contour.png', dpi=500, bbox_inches='tight')  # Save the figure
    plt.show()
    plt.clf()

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, Tangential_velocity, cmap="coolwarm",
                  shading='gouraud')  # /100 is for converting cm/sec to m/sec
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')

    plt.ylabel('z (cm)')

    plt.savefig('tangential_velocity_contour.png', dpi=500, bbox_inches='tight')  # Save the figure

    plt.clf()


def average_axial_velocity_at_each_cell():

    dict_v_z = clean_axial_velocity()
    # average the v_r at each cell
    average_v_z_component = {key: np.mean(values) if values else 0 for key, values in dict_v_z.items()}
    Axial_velocity_component = np.array(list(average_v_z_component.values()))

    return average_v_z_component, Axial_velocity_component


def average_radial_velocity_at_each_cell():
    dict_r_z = clean_radial_velocity()
    # average the v_r at each cell
    average_v_r_component = {key: np.mean(values) if values else 0 for key, values in dict_r_z.items()}
    Radial_velocity_component = np.array(list(average_v_r_component.values()))

    return average_v_r_component, Radial_velocity_component

def clean_axial_velocity():
    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    key_list = []
    velocity = []
    for i in range(number_of_points - 1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # To remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)
            key_list.append(key)
            velocity.append(v_z_mid[i] / 100)
            #print(key)
            #print(v_z_mid[i] / 100)
    #print(key_list)
    clean_indices=[i for i in range(1,len(key_list)-1) if key_list[i] != key_list[i-1] if key_list[i] != key_list[i+1]]
    velocity_raw = velocity
    clean_key = [key_list[i] for i in clean_indices]
    velocity = [velocity[i] for i in clean_indices]
    #print(velocity)
    plt.hist(velocity_raw, bins=1000,density=True, alpha=0.3, color='blue', label="raw")
    plt.hist(velocity, bins=1000,density=True, alpha=0.3,  color='red', label="clean")
    plt.xlim(-2,2)
    plt.legend()
    #plt.show()
    plt.clf()

    clean_dict_axial_velocity = {}

    for i in key_cell_index:
        clean_dict_axial_velocity[i] = []

    for idx, vel in zip(clean_key, velocity):

        clean_dict_axial_velocity[idx].append(vel)
    #print(clean_dict_axial_velocity)
    return clean_dict_axial_velocity

def clean_radial_velocity():
    v_r_mid, v_theta_mid, v_z_mid = midpoints_velocity()
    r_mid, theta_mid, z_mid = calculate_the_midpoints()
    rr, zz = mesh()

    key_cell_index = []
    for i in range(0, nr - 1):
        for j in range(0, nz - 1):
            key_cell_index.append((i, j))

    key_list = []
    velocity = []
    for i in range(number_of_points - 1):
        point_to_find = (r_mid[i], z_mid[i])
        if r_mid[i] < radius:  # To remove points outside of domain based on r, I did not remove extra z yet.
            key = find_cell_index([rr, zz], point_to_find)
            key_list.append(key)
            velocity.append(v_r_mid[i] / 100)
            #print(key)
            #print(v_z_mid[i] / 100)
    #print(key_list)
    clean_indices=[i for i in range(1,len(key_list)-1) if key_list[i] != key_list[i-1] if key_list[i] != key_list[i+1]]
    velocity_raw = velocity
    clean_key = [key_list[i] for i in clean_indices]
    velocity = [velocity[i] for i in clean_indices]
    #print(velocity)
    plt.hist(velocity_raw, bins=1000,density=True, alpha=0.3, color='blue', label="raw")
    plt.hist(velocity, bins=1000,density=True, alpha=0.3,  color='red', label="clean")
    plt.xlim(-2,2)
    plt.legend()
    plt.show()
    plt.clf()

    clean_dict_radial_velocity = {}

    for i in key_cell_index:
        clean_dict_radial_velocity[i] = []

    for idx, vel in zip(clean_key, velocity):

        clean_dict_radial_velocity[idx].append(vel)
    #print(clean_dict_axial_velocity)
    return clean_dict_radial_velocity
def plot_axial_velocity():

    center_point = write_cell_center()
    num_occurrence = new_number_of_occurrence_at_each_cell()
    average_v_z_component, Axial_velocity = average_axial_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Axial_velocity[i] = 0

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    print(len(center_x))
    print(len(Axial_velocity))
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, Axial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec #,vmin=-0.3, vmax=0.27
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')

    plt.ylabel('z (cm)')

    plt.savefig('axial_velocity_contour.png', dpi=500, bbox_inches='tight')  # Save the figure
    plt.show()

    plt.clf()
def plot_radial_velocity():

    center_point = write_cell_center()
    num_occurrence = new_number_of_occurrence_at_each_cell()
    average_v_r_component, Radial_velocity = average_radial_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1]+20)
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])
        num_occurrence = np.delete(num_occurrence, reversed_vector[i])

    for i in range(0, len(num_occurrence)):
        if num_occurrence[i] < min_occurrence:
            Radial_velocity[i] = 0

    # Create a Triangulation
    triang = Triangulation(center_x, center_y)
    print(len(center_x))
    print(len(Radial_velocity))
    # Create a filled contour plot using plt.tripcolor
    plt.tripcolor(triang, Radial_velocity, cmap="coolwarm",
                            shading='gouraud')  # /100 is for converting cm/sec to m/sec #,vmin=-0.3, vmax=0.27
    cbar = plt.colorbar()
    plt.xticks([0, 5, 10])
    plt.gca().set_aspect('equal', adjustable='box')
    # Set labels and title
    plt.xlabel('r (cm)')

    plt.ylabel('z (cm)')

    plt.savefig('axial_velocity_contour.png', dpi=500, bbox_inches='tight')  # Save the figure
    plt.show()

    plt.clf()

def new_number_of_occurrence_at_each_cell():

    dict_v_z = clean_axial_velocity()
    distribution_dict = {}
    for key, vector in dict_v_z.items():
        # Count the total number of objects in the vector
        total_count = len(vector)
        # Store the total count in the new dictionary
        distribution_dict[key] = total_count

    num_occurrence = list(distribution_dict.values())

    #print(num_occurrence)

    return num_occurrence

def output_velocity():

    # Extracting velocities and write in a and Excel file
    center_point = write_cell_center()
    average_v_z_component, Axial_velocity = average_axial_velocity_at_each_cell()
    average_v_r_component, Radial_velocity = average_radial_velocity_at_each_cell()
    center_x = []
    center_y = []
    cell_to_remove = []

    for i in range(0, (len(center_point))):
        if center_point[i][0] < radius and z_min_tank < center_point[i][1] < z_max_tank:
            center_x.append(center_point[i][0])
            center_y.append(center_point[i][1])
        else:
            cell_to_remove.append(i)

    reversed_vector = cell_to_remove[::-1]

    for i in range(0, len(reversed_vector)):
        Axial_velocity = np.delete(Axial_velocity, reversed_vector[i])
        Radial_velocity = np.delete(Radial_velocity, reversed_vector[i])


    # Specify the file path
    file_path = "output_velocity_experiment.txt"

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the header line
        file.write("r z radial_velocity  axial_velocity\n")

        # Iterate over the elements of the vectors
        for i in range(len(center_x)):
            # Write the elements to the file
            file.write(f"{center_x[i]} {center_y[i]} {Radial_velocity[i]} {Axial_velocity[i]}\n")
#####################################################################

x_pred, y_pred, z_pred = predict_position()
#plot_occurrence_distribution()
#plot_velocity()
#plot_sd()
output_velocity()
#output_stresses()
#fluctuating_velocity()
#plot_vertex()
#plot_cfd_axial_profile()
#plot_cfd_stresses()
#plot_superimposed_contour_and_vector()
#average_velocity_at_each_cell()
#plot_stresses()
#assign_velocity_to_cells()
#midpoints_velocity()
#plot_event_distribution()
#plot_event_distribution()
#plot_difference_between_event_occurrence()
#data_for_histogram_axial_velocity()
#assign_velocity_to_cells()
#data_for_scatter_plot()
#clean_axial_velocity()
plot_axial_velocity()
plot_radial_velocity()


