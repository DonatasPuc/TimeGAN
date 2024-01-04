import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def load_data(feat_number):
    """
    Load the frequency data from a .mat file.
    """
    file_path = f"data/gear_signals/Feat{feat_number}/1 bandymas/75/3000/5.mat"
    data = scipy.io.loadmat(file_path)
    frequencies = data['Data'].flatten()
    return frequencies

def plot_data(frequencies, interval, sampling_rate, feat_number):
    """
    Plot and save the frequency data for a specified time interval.
    """
    num_points = int(sampling_rate * interval)
    data_subset = frequencies[:num_points]
    time = np.linspace(0, interval, num_points)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, data_subset)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Frequency vs Time ({interval} second interval) - Feature {feat_number}')
    plt.ylim(-0.4, 0.4)
    plt.grid(True)
    
    if not os.path.exists('plots_raw'):
        os.makedirs('plots_raw')
    filename = f'plots_raw/feature{feat_number}_{interval}s_interval.png'
    plt.savefig(filename)
    plt.close()

def plot_combined(features, interval, sampling_rate):
    """
    Combine and plot data from multiple features.
    """
    plt.figure(figsize=(10, 6))
    for feat_number in features:
        frequencies = load_data(feat_number)
        num_points = int(sampling_rate * interval)
        data_subset = frequencies[:num_points]
        time = np.linspace(0, interval, num_points)
        plt.plot(time, data_subset, label=f'Feature {feat_number}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Combined Frequency vs Time ({interval} second interval)')
    plt.ylim(-0.4, 0.4)
    plt.legend()
    plt.grid(True)
    
    if not os.path.exists('plots_raw'):
        os.makedirs('plots_raw')
    feature_str = '_'.join([f'feat{feat}' for feat in features])
    filename = f'plots_raw/combined_{feature_str}_{interval}s_interval.png'
    plt.savefig(filename)
    plt.close()

# Example usage
sampling_rate = 256000 / 5  # Total number of measurements divided by total duration in seconds

features = [0, 3, 6]  # Example feature numbers
# for feat in features:
#     frequencies = load_data(feat)
#     plot_data(frequencies, 1, sampling_rate, feat)  # For 1 second interval
#     plot_data(frequencies, 0.0005, sampling_rate, feat)  # For 0.0005 second interval

# Plotting combined graph for specified features
# plot_combined(features, 1, sampling_rate)  # Combined plot for 1 second interval
plot_combined(features, 0.005, sampling_rate)  # Combined plot for 0.0005 second interval
