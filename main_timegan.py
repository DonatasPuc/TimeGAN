"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import numpy as np
import warnings
from datetime import datetime
from log_config import setup_logging
warnings.filterwarnings("ignore")


# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, save_ndarray_to_mat
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization


def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, energy or gear_signals
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  # Initialize logging
  args = parser.parse_args()
  setup_logging(args.logfile)

  logging.info("Started with the following arguments: %s", vars(args))

  ## Data loading
  if args.data_name in ['stock', 'energy']:
    ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
  elif args.data_name == 'gear_signals':
    # Set number of samples and its dimensions
    ori_data = real_data_loading("gear_signals", args.seq_len, base_path="data", percentage=args.dataset_percentage, feat_numbers=[args.feature_number], exp_number="1", torque="75", rpm="3000")
    
  logging.info(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
      
  generated_data = timegan(ori_data, parameters, 'models', 'models', 100)
  logging.info('Finish Synthetic Data Generation')

  save_ndarray_to_mat(generated_data, f'feat3_25prc_75_3000_synth_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  # 1. Discriminative Score
  discriminative_score = list()
  for _ in range(args.metric_iteration):
    temp_disc = discriminative_score_metrics(ori_data, generated_data)
    discriminative_score.append(temp_disc)
      
  metric_results['discriminative'] = np.mean(discriminative_score)
      
  # 2. Predictive score
  predictive_score = list()
  for tt in range(args.metric_iteration):
    temp_pred = predictive_score_metrics(ori_data, generated_data)
    predictive_score.append(temp_pred)   
      
  metric_results['predictive'] = np.mean(predictive_score)     
          
  # 3. Visualization (PCA and tSNE)
  visualization(ori_data, generated_data, 'pca', plot_synthetic=True)
  visualization(ori_data, generated_data, 'tsne', plot_synthetic=True)
  
  ## Print discriminative and predictive scores
  logging.info(metric_results)

  return ori_data, generated_data, metric_results


# def print_array_info(array, name):
#     print(f"Information about {name}:")
#     print(f"  Shape: {array.shape}")
#     print(f"  Data Type: {array.dtype}")
#     print(f"  Size: {array.size}")
#     print(f"  Number of Dimensions: {array.ndim}")
#     print(f"  Item Size: {array.itemsize} bytes")
#     print(f"  Minimum Value: {np.min(array)}")
#     print(f"  Maximum Value: {np.max(array)}")
#     print(f"  Mean: {np.mean(array)}")
#     print(f"  Standard Deviation: {np.std(array)}")
#     print(" ")


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name', 
      choices=['sine','stock','energy','gear_signals', 'synth_gear_signals'],
      default='stock',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=10,
      type=int)
  parser.add_argument(
      '--logfile',
      help='the location of the log file to which logs will be written',
      default='timegan.log',
      type=str)
  parser.add_argument(
      '--feature_number',
      help='the location of the log file to which logs will be written',
      choices=['Feat0','Feat1','Feat2','Feat3','Feat4','Feat5','Feat6'],
      default='Feat0',
      type=str)
  parser.add_argument(
      '--dataset_percentage',
      help='load only the specified percentage of data from the original dataset',
      default=10,
      type=int)
  
  args = parser.parse_args()
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)