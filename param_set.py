# Number of inner loop of nested cv
cv_num_valid = 5
# Number of outer loop of nested cv(not LOSO)
cv_num_test = 10
# Calculate FC data and make lists of labels
# After .bf files of data and labelss are saved in ource_dir, please set it to False
calculate_fc_FLAG = True
# If this is set to be True,  the trained models and performance indices are saved
save_FLAG = True
# Relative path to save data and labels intermidiate files from work space
source_dir = './sources'
# Relative path to save trained models and performance indices
output_dir = './results'

# Parameters for generating intermidiate files
preparation_params = {
    # Relative path of phenotypic data file
    'label_file_path': './Phenotypic_V1_0b_preprocessed1.csv',
    # Relative path of each ROI timeseries files
    # Divide the path into 3 parts; 'site_id+subj_id' part and back and forth of it
    'path_structure': ['./Outputs/cpac/filt_global/rois_ho/','site_id+subj_id', '_rois_ho.1D']

}

# Parameters for neural network training
nn_params = {
    'device': 'cuda',
    # Batch size of pre-training
    'batch_size': 500,
    'batch_size_rev': 500,
    # Number of objective classes
    'num_classes': 2,
    # Number of over clustering classes
    'oc_class_num': 10,
    # Number of nodes of the layer just before the output
    'prefinal_num': 100,
    # Weight of marginal entropy in IIC loss
    # alpha_ad: same label pair
    'alpha_ad': 5.0,
    # alpha_rev: different label pair
    'alpha_rev': 5.0,
    # alpha_oc: overclustering of same label pair
    'alpha_oc': 5.0,
    # Upper limit of training epoch
    # To stop training by early stopping, this parameter is set large
    'total_epoch': 1000,
    # Early stopping epoch
    # When validation loss does not decreased for this epoch, stop training
    'es_epoch': 10,
    'learning_rate': 1e-4,
    # Parameters for transfer learning and simple MLP training
    'total_epoch_cl': 10000,
    'batch_size_cl': 50,
    'es_epoch_cl': 50,
}

# Searched hyper parameters for SVM
tuning_params_svm = {
    'C': [10 **i for i in range(-4,3)], 
    'gamma': [10 **i for i in range(-4,3)]
}

# Searched hyper parameters for Random Forest
tuning_params_rf = {
    'n_estimators': [i for i in range(5,20+1, 5)],
    'max_depth': [i for i in range(1, 5+1)]
}

def training_params():
    params = {
        'cv_num': cv_num_valid,
        'cv_num_test': cv_num_test,
        'save_FLAG': save_FLAG,
        'source_dir': source_dir,
        'output_dir': output_dir,
        'nn_params': nn_params,
        'tuning_params_svm': tuning_params_svm,
        'tuning_params_rf': tuning_params_rf
    }
    return params
