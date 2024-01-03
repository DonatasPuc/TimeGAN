# Used for testing of data visualization

import numpy as np
from data_loading import real_data_loading
from metrics.visualization_metrics import visualization


ori_data   = real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat0"], exp_number="1", torque="75", rpm="3000")
ori_data_2 = real_data_loading("gear_signals", (int) (2560), base_path="data", feat_numbers=["Feat6"], exp_number="2", torque="75", rpm="3000")


visualization(ori_data, ori_data_2, 'pca', plot_synthetic=True)
visualization(ori_data, ori_data_2, 'tsne', plot_synthetic=True)