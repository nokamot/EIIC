import os
import random
import numpy as np
import torch

from param_set import training_params
from data_preparation import ts_processing
from loso_cv import loso_nested_cv, nested_cv

# Fix random seed
SEED_VALUE = 0 
os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)

params = training_params()

# Generate intermidiate files
if params['calculate_fc_FLAG']:
    ts_processing(params['source_dir'], params['preparation_params'])

# Leave one site out cv
accs_transfer, accs_classifier, accs_svm, accs_rf = loso_nested_cv(params)

# Normal nested cv
#accs_transfer, accs_classifier, accs_svm, accs_rf = nested_cv(params)

# Print each site result
"""
print("each_sites_results\n"+
      "contrastive_nn:\n")
print(accs_transfer)
print("\n"+
      "non_contrastive_nn:\n")
print(accs_classifier)
print("\n"+
      "svm:\n")
print(accs_svm)
print("\n"+
      "random_forest:\n")
print(accs_rf)
print("\n"
     )
"""

# Print mean accuracy
print("mean_result\n"+
      "contrastive_nn:{:.3f}\n".format(np.mean(accs_transfer))+
      "non_contrastive_nn:{:.3f}\n".format(np.mean(accs_classifier))+
      "svm:{:.3f}\n".format(np.mean(accs_svm))+
      "random_forest:{:.3f}".format(np.mean(accs_rf))
     )
