import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from data_preparation import prepare_data
from nn_training import train_transfer, train_classifier
from scikit_classifier import train_svm, train_rf, prediction_matrix
from test_evaluation import matrix_from_models, matrix_from_models_cl
from utils import save_models, model_measurements, save_result_csv


# Leave one site out
def loso_nested_cv(params):
    
    # The number of validation split
    cv_num = params['cv_num']
    # Data and labels 
    data, hc_ad, site_id_, split_ref_, ref = prepare_data(params['source_dir'])
    
    # Numpy array for ML models' performance indices
    measurements_transfer = np.zeros(shape=[len(ref), 5])
    measurements_classifier = np.zeros(shape=[len(ref), 5])
    measurements_svm = np.zeros(shape=[len(ref), 5])
    measurements_rf = np.zeros(shape=[len(ref), 5])

    # List of accuracy for final output
    accs_transfer = []
    accs_classifier = []
    accs_svm = []
    accs_rf = []

    for site in range(0,len(ref)):

        # List of standard scalers for each validation split
        scalers = []

        # List of trained models
        trained_transfer = []
        trained_classifier = []
        epochs = []

        # Split data
        x_train_ = data[np.where(site_id_!=site)[0]]
        x_test = data[np.where(site_id_==site)[0]]
        labels_train_ = hc_ad[np.where(site_id_!=site)[0]]
        labels_test = hc_ad[np.where(site_id_==site)[0]]
    
        # Split keeping the ratio of acquisition sites and ASD/TC simultaneously
        split_ref = split_ref_[np.where(site_id_!=site)[0]]

        skf = StratifiedKFold(n_splits=cv_num, random_state=0, shuffle=True)
        
        for cv_iteration, (train_index, valid_index) in enumerate(skf.split(labels_train_, split_ref)):
            epochs_cv = []
            # Split data
            x_train, x_valid = x_train_[train_index], x_train_[valid_index]
            labels_train, labels_valid = labels_train_[train_index], labels_train_[valid_index]
            
            # Standardize input
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            
            scalers.append(scaler)
            
            # Train EIIC model
            model_trained_transfer = train_transfer(x_train, labels_train, x_valid, labels_valid, params['nn_params'])
            trained_transfer.append(model_trained_transfer)
            
            # Train simple MLP without contrastive learning
            model_trained_classifier = train_classifier(x_train, labels_train, x_valid, labels_valid, params['nn_params'])
            trained_classifier.append(model_trained_classifier)
        
        # Calculate confusion matrix from trained model
        matrix = matrix_from_models(trained_transfer, scalers, x_test, labels_test, params['nn_params']['num_classes'], params['nn_params']['device'])
        # Calculate performance indices from confusion matrix
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_transfer.append(acc)
        measurements_transfer[site,:] = [acc, recall, specificity, ppv, npv]
        
        
        matrix = matrix_from_models_cl(trained_classifier, scalers, x_test, labels_test, params['nn_params']['num_classes'], params['nn_params']['device'])
    
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_classifier.append(acc)
        measurements_classifier[site,:] = [acc, recall, specificity, ppv, npv]
        
        # PCA dimensionality reduction for SVM and RF
        pca = PCA(n_components = params['nn_params']['prefinal_num'])

        # Get the transformation of PCA from data excepting test data
        pca.fit(x_train_)

        # Transform data
        x_train_sc = pca.transform(x_train_)
        x_test_sc = pca.transform(x_test)
        
        # Train SVM and RF
        svm = train_svm(x_train_sc, labels_train_, skf.split(labels_train_, split_ref), params['tuning_params_svm'])
        rf = train_rf(x_train_sc, labels_train_, skf.split(labels_train_, split_ref), params['tuning_params_rf'])
        
        # Calculate confusion matrix and performance indices
        matrix = prediction_matrix(x_test_sc, labels_test, svm, params['nn_params']['num_classes'])
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_svm.append(acc)
        measurements_svm[site,:] = [acc, recall, specificity, ppv, npv]
        
        matrix = prediction_matrix(x_test_sc, labels_test, rf, params['nn_params']['num_classes'])
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_rf.append(acc)
        measurements_rf[site,:] = [acc, recall, specificity, ppv, npv]
        
        # Save models
        if params['save_FLAG']:
            models = {
                'nn_scalers': scalers,
                'pca': pca,
                'EIIC_models': trained_transfer,
                'MLP_models': trained_classifier,
                'svm_model': svm,
                'rf_model': rf
            }
            save_models(models, params['output_dir'], ref[site])

        print("Finished site: "+ref[site])
    
    # Output the performance indices as xlsx
    if params['save_FLAG']:
        models = {
            'EIIC_models': measurements_transfer,
            'MLP_models': measurements_classifier,
            'svm_model': measurements_svm,
            'rf_model': measurements_rf
        }
        save_result_csv(models, ref, ['acc', 'recall', 'specificity', 'ppv', 'npv'], params['output_dir'])
        
    # Return accuracies for output
    return accs_transfer, accs_classifier, accs_svm, accs_rf
        
# Nested cv
def nested_cv(params):
    
    # The number of validation split
    cv_num = params['cv_num']
    # The number of test split
    cv_num_test = params['cv_num_test']
    # Data and labels
    data, hc_ad, site_id_, split_ref_, ref = prepare_data(params['source_dir'])
    
    # Numpy array for ML models' performance indices
    measurements_transfer = np.zeros(shape=[cv_num_test, 5])
    measurements_classifier = np.zeros(shape=[cv_num_test, 5])
    measurements_svm = np.zeros(shape=[cv_num_test, 5])
    measurements_rf = np.zeros(shape=[cv_num_test, 5])

    # List of accuracy for final output
    accs_transfer = []
    accs_classifier = []
    accs_svm = []
    accs_rf = []

    skf_ = StratifiedKFold(n_splits=cv_num_test, random_state=0, shuffle=True)

    # Split keeping the ratio of acquisition sites and ASD/TC simultaneously
    for cv_iteration_, (train_index_, test_index) in enumerate(skf_.split(data, split_ref_)):

        # List of standard scalers for each validation split
        scalers = []

        # List of trained models
        trained_transfer = []
        trained_classifier = []
        epochs = []

        # Split data
        x_train_, x_test = data[train_index_], data[test_index]
        labels_train_, labels_test = hc_ad[train_index_], hc_ad[test_index]
        split_ref = split_ref_[train_index_]
        
        skf = StratifiedKFold(n_splits=cv_num, random_state=0, shuffle=True)
        
        # Split keeping the ratio of acquisition sites and ASD/TC simultaneously
        for cv_iteration, (train_index, valid_index) in enumerate(skf.split(x_train_, split_ref)):
            epochs_cv = []
            # Split data
            x_train, x_valid = x_train_[train_index], x_train_[valid_index]
            labels_train, labels_valid = labels_train_[train_index], labels_train_[valid_index]
            
            # Standardize input
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)
            
            scalers.append(scaler)
            
            # Train EIIC model
            model_trained_transfer = train_transfer(x_train, labels_train, x_valid, labels_valid, params['nn_params'])
            trained_transfer.append(model_trained_transfer)
            
            # Train simple MLP without contrastive learning
            model_trained_classifier = train_classifier(x_train, labels_train, x_valid, labels_valid, params['nn_params'])
            trained_classifier.append(model_trained_classifier)
        
        # Calculate confusion matrix from trained model
        matrix = matrix_from_models(trained_transfer, scalers, x_test, labels_test, params['nn_params']['num_classes'], params['nn_params']['device'])
        
        # Calculate performance indices from confusion matrix
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_transfer.append(acc)
        measurements_transfer[cv_iteration_,:] = [acc, recall, specificity, ppv, npv]
            
        matrix = matrix_from_models_cl(trained_classifier, scalers, x_test, labels_test, params['nn_params']['num_classes'], params['nn_params']['device'])
    
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_classifier.append(acc)
        measurements_classifier[cv_iteration_,:] = [acc, recall, specificity, ppv, npv]
        
        # PCA dimensionality reduction for SVM and RF
        pca = PCA(n_components = params['nn_params']['prefinal_num'])

        # Get the transformation of PCA from data excepting test data
        pca.fit(x_train_)

        # Transform data
        x_train_sc = pca.transform(x_train_)
        x_test_sc = pca.transform(x_test)
        
        # Train SVM and RF
        svm = train_svm(x_train_sc, labels_train_, skf.split(x_train_, split_ref), params['tuning_params_svm'])
        rf = train_rf(x_train_sc, labels_train_, skf.split(x_train_, split_ref), params['tuning_params_rf'])
        
        # Calculate confusion matrix and performance indices
        matrix = prediction_matrix(x_test_sc, labels_test, svm, params['nn_params']['num_classes'])
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_svm.append(acc)
        measurements_svm[cv_iteration_,:] = [acc, recall, specificity, ppv, npv]
        
        matrix = prediction_matrix(x_test_sc, labels_test, rf, params['nn_params']['num_classes'])
        acc, recall, specificity, ppv, npv = model_measurements(matrix)
        accs_rf.append(acc)
        measurements_rf[cv_iteration_,:] = [acc, recall, specificity, ppv, npv]
        
        # Save models
        if params['save_FLAG']:
            models = {
                'nn_scalers': scalers,
                'pca': pca,
                'transfer_models': trained_transfer,
                'classifier_models': trained_classifier,
                'svm_model': svm,
                'rf_model': rf
            }
            save_models(models, params['output_dir'], str(cv_iteration_))
        
    # Output the performance indices as xlsx
    if params['save_FLAG']:
        models = {
            'transfer_models': measurements_transfer,
            'classifier_models': measurements_classifier,
            'svm_model': measurements_svm,
            'rf_model': measurements_rf
        }
        save_result_csv(models, ref, ['acc', 'recall', 'specificity', 'ppv', 'npv'], params['output_dir'])

    # Return accuracies for output
    return accs_transfer, accs_classifier, accs_svm, accs_rf
