"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import pandas as pd
import scipy.io


def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    
def load_specific_data(base_path, feat_number, torque, rpm):
    """
    Load specific dataset based on feature number, torque, and rpm.

    Args:
      - base_path: The base path where the datasets are located.
      - feat_number: The feature number (e.g., Feat0, Feat1, etc.).
      - torque: The torque value.
      - rpm: The rpm value.

    Returns:
      - data: Loaded dataset.
    """
    file_path = f"{base_path}/gear_signals/{feat_number}/1 bandymas/{torque}/{rpm}/5.mat"
    data = scipy.io.loadmat(file_path)
    return data['Data']

def load_multiple_features(base_path, feat_numbers, exp_number, torque, rpm):
    """
    Load multiple datasets based on a list of feature numbers, torque, and rpm.

    Args:
      - base_path: The base path where the datasets are located.
      - feat_numbers: A list of feature numbers (e.g., ['Feat0', 'Feat1', ...]).
      - torque: The torque value.
      - rpm: The rpm value.

    Returns:
      - data: NumPy ndarray containing loaded datasets in different columns.
    """
    all_data = []

    for feat_number in feat_numbers:
        file_path = f"{base_path}/gear_signals/{feat_number}/{exp_number} bandymas/{torque}/{rpm}/5.mat"
        data = scipy.io.loadmat(file_path)['Data']
        all_data.append(data)

    combined_data = np.hstack(all_data)
    return combined_data

def real_data_loading (data_name, seq_len, base_path=None, feat_numbers=None, exp_number=None, torque=None, rpm=None):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    - base_path: The base directory for gear_signals data.
    - feat_number: The feature number for gear_signals data.
    - torque: The torque value for gear_signals data.
    - rpm: The rpm value for gear_signals data.
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock', 'energy', 'gear_signals']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'gear_signals':
     # Ensure feat_numbers is a list
      if not isinstance(feat_numbers, list):
          feat_numbers = [feat_numbers]
      ori_data = load_multiple_features(base_path, feat_numbers, exp_number, torque, rpm)
    
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data