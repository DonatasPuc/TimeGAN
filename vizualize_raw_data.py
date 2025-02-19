import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def load_data(path):
    """
    Load the frequency data from a .mat file.
    """
    data = scipy.io.loadmat(path)
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

def plot_combined(features, interval, sampling_rate, additional_features=None, offset=0):
    """
    Combine and plot data from multiple features with an option to start at an offset in seconds.
    additional_features should be a list of dictionaries, each containing 'path' and 'label' keys.
    Example: [{'path': 'path/to/feature1.mat', 'label': 'Feature 1'}, {'path': 'path/to/feature2.mat', 'label': 'Feature 2'}]
    """
    plt.figure(figsize=(10, 6))
    start_index = int(sampling_rate * offset)
    end_index = start_index + int(sampling_rate * interval)

    def safe_load_data(path):
        if os.path.exists(path):
            return scipy.io.loadmat(path)['Data'].flatten()
        else:
            print(f"Warning: File not found at {path}")
            return None

    for feat_number in features:
        path = f"data/gear_signals/Feat{feat_number}/1 bandymas/75/3000/5.mat"
        data = safe_load_data(path)
        if data is not None:
            data_subset = data[start_index:end_index]
            time = np.linspace(offset, offset + interval, len(data_subset))
            plt.plot(time, data_subset, label=f'Feature {feat_number}')

    if additional_features:
        for feature in additional_features:
            additional_data = safe_load_data(feature['path'])
            if additional_data is not None:
                data_subset = additional_data[start_index:end_index]
                time = np.linspace(offset, offset + interval, len(data_subset))
                plt.plot(time, data_subset, label=feature['label'])

    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Combined Frequency vs Time (Interval: {interval} sec, Offset: {offset} sec)')
    plt.ylim(-0.4, 0.4)
    plt.legend()
    plt.grid(True)

    if not os.path.exists('plots_raw'):
        os.makedirs('plots_raw')
    feature_str = '_'.join([f'feat{feat}' for feat in features])
    filename = f'plots_raw/combined_{feature_str}_{interval}s_{offset}s_offset.png'
    plt.savefig(filename)
    plt.close()

# Example usage
sampling_rate = 256000 / 5  # Total number of measurements divided by total duration in seconds

#features = [3]  # Example feature numbers
# for feat in features:
#     frequencies = load_data(feat)
#     plot_data(frequencies, 1, sampling_rate, feat)  # For 1 second interval
#     plot_data(frequencies, 0.0005, sampling_rate, feat)  # For 0.0005 second interval

# Plotting combined graph for specified features
# plot_combined(features, 1, sampling_rate)  # Combined plot for 1 second interval
generated_data_path = 'generated_data/feat3_25prc_75_3000_synth_20240110_013014.mat'
# plot_combined([3], 0.0025, sampling_rate, offset=1, additional_features=[{'path': generated_data_path, 'label': 'synthetic'}])
# plot_combined([3], 0.0055, sampling_rate, offset=1, additional_features=[{'path': generated_data_path, 'label': 'synthetic'}])
# plot_combined([3], 0.025, sampling_rate, offset=1, additional_features=[{'path': generated_data_path, 'label': 'synthetic'}])
# plot_combined([3], 0.25, sampling_rate, offset=1, additional_features=[{'path': generated_data_path, 'label': 'synthetic'}])

# plot_combined([3], 5, sampling_rate, offset=0, additional_features=[{'path': generated_data_path, 'label': 'synthetic'}])
plot_combined([3], 0.0026, sampling_rate, offset=1, additional_features=[{'path': 'generated_data/feat3_25prc_75_3000_synth_20240110_083803_multi.mat', 'label': 'synthetic'}])
plot_combined([3], 0.0056, sampling_rate, offset=1, additional_features=[{'path': 'generated_data/feat3_25prc_75_3000_synth_20240110_083803_multi.mat', 'label': 'synthetic'}])
plot_combined([3], 0.99, sampling_rate, offset=1, additional_features=[{'path': 'generated_data/feat3_25prc_75_3000_synth_20240110_083803_multi.mat', 'label': 'synthetic'}])
plot_combined([3], 4.99, sampling_rate, offset=0, additional_features=[{'path': 'generated_data/feat3_25prc_75_3000_synth_20240110_083803_multi.mat', 'label': 'synthetic'}])
