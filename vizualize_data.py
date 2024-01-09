# Used for testing of data visualization

from datetime import datetime
import numpy as np
import data_loading as dl
from metrics.visualization_metrics import visualization
from timegan import timegan


# ori_data   = dl.real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat0"], exp_number="1", torque="75", rpm="3000")
# ori_data_2 = dl.real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat6"], exp_number="2", torque="75", rpm="3000")


# visualization(ori_data, ori_data_2, 'pca', plot_synthetic=True)
# visualization(ori_data, ori_data_2, 'tsne', plot_synthetic=True)

percentage = 25
feature_num = 3
seq_len = 32

ori_data   = dl.real_data_loading("gear_signals", seq_len, base_path="data", percentage=percentage, feat_numbers=[f"Feat{feature_num}"], exp_number="1", torque="75", rpm="3000")

# loading original data as synthetic
# rec_ori_data = dl.reconstruct_data(np.asarray(ori_data), seq_len)
# dl.save_ndarray_to_mat(rec_ori_data, f'feat{feature_num}_{percentage}prc_75_3000')
# synth_data = dl.real_data_loading("synth_gear_signals", seq_len, synthetic_data_file=f'feat{feature_num}_{percentage}prc_75_3000')

# generating synthetic
# Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 1000
parameters['batch_size'] = 128
parameters['data_name'] = 'gear_signals'
parameters['dataset_percentage'] = percentage
parameters['seq_len'] = seq_len
parameters['feature_number'] = 'Feat3'
parameters['do_training'] = False
parameters['model_dir'] = 'models_feat3_0109_2117_seq_32'

synth_data = timegan(ori_data, parameters, parameters['model_dir'], parameters['model_dir'], 100, parameters['do_training'])

save_file = f'feat{feature_num}_{percentage}prc_75_3000_synth_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
dl.save_ndarray_to_mat(dl.reconstruct_data(np.asarray(synth_data), seq_len), save_file)
dl.save_ndarray_to_mat(synth_data, save_file + '_multi')


# visualization(ori_data, synth_data, 'pca', plot_synthetic=True)
# visualization(ori_data, synth_data, 'tsne', plot_synthetic=True)