from training_function import pre_training, cl_training, classifier_training

import torch
import torch.nn as nn

# Contrastive learning
def train_transfer(x_train, labels_train, x_valid, labels_valid, nn_params):
    
    # Pre-training
    model_pre_trained, final_epoch = pre_training(x_train, labels_train, x_valid, labels_valid, nn_params)
    
    model_pre_trained.to('cpu')
    # Transfer learning
    for param in model_pre_trained.parameters():
        param.requires_grad = False

    model_pre_trained.dense3_cl = nn.Linear(nn_params['prefinal_num'], nn_params['num_classes'])
    
    model_pre_trained.to(nn_params['device'])
    
    model_trained = cl_training(model_pre_trained, x_train, labels_train, x_valid, labels_valid, nn_params)
    
    return model_trained

# Non-contrastive learning
def train_classifier(x_train, labels_train, x_valid, labels_valid, nn_params):
    
    
    model_trained = classifier_training(x_train, labels_train, x_valid, labels_valid, nn_params)
    
    return model_trained
