# Used for testing of data visualization

import numpy as np
import data_loading as dl
from metrics.visualization_metrics import visualization


# ori_data   = dl.real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat0"], exp_number="1", torque="75", rpm="3000")
# ori_data_2 = dl.real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat6"], exp_number="2", torque="75", rpm="3000")


# visualization(ori_data, ori_data_2, 'pca', plot_synthetic=True)
# visualization(ori_data, ori_data_2, 'tsne', plot_synthetic=True)

percentage = 25
feature_num = 3
seq_len = 256

ori_data   = dl.real_data_loading("gear_signals", seq_len, base_path="data", percentage=percentage, feat_numbers=[f"Feat{feature_num}"], exp_number="1", torque="75", rpm="3000")

rec_ori_data = dl.reconstruct_data(np.asarray(ori_data), seq_len)
dl.save_ndarray_to_mat(rec_ori_data, f'feat{feature_num}_{percentage}prc_75_3000')

synth_data = dl.real_data_loading("synth_gear_signals", seq_len, synthetic_data_file=f'feat{feature_num}_{percentage}prc_75_3000')


visualization(ori_data, ori_data, 'pca', plot_synthetic=True)
visualization(ori_data, synth_data, 'tsne', plot_synthetic=True)