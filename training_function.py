from iic_loss import IID_loss
from nn_models import NetIIC, NetClassify

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

# Pre-training
def train_pre(total_epoch, model, train_loader, train_loader_rev, valid_loader, optimizer, es, alpha_ad, alpha_oc, alpha_rev, device):
    
    best_loss = 1e4
    es_count = 0
    
    # Function for oscillation of learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2)
    
    for epoch in range(total_epoch):

        model.train()
        
        # Training for same labels pairs
        for batch_idx, (data, data_perturb) in enumerate(train_loader):

            # Update learning rate
            scheduler.step()

            data = data.to(device)
            data_perturb = data_perturb.to(device)

            optimizer.zero_grad()

            output, output_overclustering, _, _ = model(data)
            output_perturb, output_perturb_overclustering, _, _ = model(data_perturb)

            loss1 = IID_loss(output, output_perturb, alpha_ad)
            
            loss2 = IID_loss(output_overclustering,
                             output_perturb_overclustering, alpha_oc)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()
            
        # Training for different labels pairs
        for batch_idx, (data, data_) in enumerate(train_loader_rev):
            
            # Update learning rate
            scheduler.step()
            
            data = data.to(device)
            data_ = data_.to(device)
            
            optimizer.zero_grad()
            
            output_rev, _, _, _ = model(data)
            _, _, output_rev_, _ = model(data_)
            
            loss1 = IID_loss(output_rev, output_rev_, alpha_rev)
            
            loss = loss1
            
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, data_) in enumerate(valid_loader):

                data = data.to(device)
                data_ = data_.to(device)

                optimizer.zero_grad()

                output, output_overclustering, _, _ = model(data)
                output_perturb, output_perturb_overclustering, _, _ = model(data_)

                loss1 = IID_loss(output, output_perturb, alpha_ad)
                loss2 = IID_loss(output_overclustering,
                                 output_perturb_overclustering, alpha_oc)
                val_loss += loss1.item() + loss2.item()
        
        # Check validation loss
        if val_loss<best_loss:
            best_loss = val_loss
            best_model = model
            best_optimizer = optimizer
            es_count = 0
        else:
            es_count += 1

        # Check early stopping
        if es_count >= es:
            #print("Early Stopping")
            break
        # Print log
        #print('Train Epoch {}:iter{} - \tLoss1: {:.6f}- \tLoss2: {:.6f}- \tLoss_total: {:.6f}'.format(
        #    epoch, batch_idx, loss1.item(), loss2.item(), loss1.item()+loss2.item()))

    return best_model, best_optimizer, best_loss, epoch


# Make data pair and model training
def pre_training(x_train, labels_train, x_valid, labels_valid, nn_params):

    input_num = x_train.shape[1]


    ind = np.where(labels_train==1)[0]
    # List of the indices of same label pair
    ind_set = []
    for i in range(len(ind)):
        for j in range(len(ind)-i-1):
            #if site_id[ind[i]]!=site_id[ind[i+1+j]]:
            ind_set.append([ind[i],ind[i+1+j]])
    ind_ = np.where(labels_train==0)[0]
    for i in range(len(ind_)):
        for j in range(len(ind_)-i-1):
            #if site_id[ind_[i]]!=site_id[ind_[i+1+j]]:
            ind_set.append([ind_[i],ind_[i+1+j]])

    # List of the indices of different label pair 
    ind_set_rev = []
    for i in (ind):
        for j in (ind_):
            ind_set_rev.append([i,j])

    # Paired indices for validation
    ind_set_valid = []
    ind = np.where(labels_valid==1)[0]
    for i in range(len(ind)):
        for j in range(len(ind)-i-1):
            #if site_id[ind[i]]!=site_id[ind[i+1+j]]:
            ind_set_valid.append([ind[i],ind[i+1+j]])
    ind_ = np.where(labels_valid==0)[0]
    for i in range(len(ind_)):
        for j in range(len(ind_)-i-1):
            #if site_id[ind_[i]]!=site_id[ind_[i+1+j]]:
            ind_set_valid.append([ind_[i],ind_[i+1+j]])

    # Make data pair from index pair
    train_data = [(Tensor(x_train[ind[0]]), Tensor(x_train[ind[1]])) for ind in ind_set]
    train_data_rev = [(Tensor(x_train[ind[0]]), Tensor(x_train[ind[1]])) for ind in ind_set_rev]

    valid_data = [(Tensor(x_valid[ind[0]]), Tensor(x_valid[ind[1]])) for ind in ind_set_valid]

    # Change list to pytorch DataLoader
    train_loader = DataLoader(train_data, batch_size=nn_params['batch_size'], shuffle=True, drop_last=True)#, num_workers=2)
    train_loader_rev = DataLoader(train_data_rev, batch_size=nn_params['batch_size_rev'], shuffle=True, drop_last=True)#, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)

    device = nn_params['device']

    model = NetIIC(input_num, nn_params['prefinal_num'], nn_params['oc_class_num'])
    model.apply(weight_init)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=nn_params['learning_rate'])

    model_trained, optimizer, val_loss, final_epoch = train_pre(
        nn_params['total_epoch'], model, train_loader, train_loader_rev, valid_loader, optimizer, nn_params['es_epoch'], nn_params['alpha_ad'], nn_params['alpha_oc'], nn_params['alpha_rev'], device)

    return model_trained, final_epoch

# Transfer learning
def train_cl(total_epoch, criterion, model, train_loader, valid_loader, optimizer, es, device):

    best_loss = 1e4
    es_count = 0
    
    # Function for oscillation of learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2)
    
    for epoch in range(total_epoch):

        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):

            # Update learning rate
            scheduler.step()
る
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            _, _, _, output_cl = model(data)
            
            ce_loss = criterion(output_cl, target)
            
            loss = ce_loss

            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):

                data = data.to(device)
                target = target.to(device)

                _, _, _, output = model(data)

                ce_loss1 = criterion(output, target)
                val_loss += ce_loss1.item()
        
        # Check validation loss
        if val_loss<best_loss:
            best_loss = val_loss
            best_model = model
            best_optimizer = optimizer
            es_count = 0
        else:
            es_count += 1

        # Check early stopping
        if es_count >= es:
            #print("Early Stopping")
            break
        # Print log
        #print('Train Epoch {}:iter{} - \tLoss1: {:.6f}- \tLoss2: {:.6f}- \tLoss_total: {:.6f}'.format(
        #    epoch, batch_idx, loss1.item(), loss2.item(), loss1.item()+loss2.item()))

    return best_model, best_optimizer, best_loss, epoch


# Make dataloader and train model
def cl_training(model_pre_trained, x_train, labels_train, x_valid, labels_valid, nn_params):

    train_data = [(Tensor(data), torch.tensor(int(target))) for data, target in zip(x_train, labels_train)]
    valid_data = [(Tensor(data), torch.tensor(int(target))) for data, target in zip(x_valid, labels_valid)]

    train_loader = DataLoader(train_data, batch_size=nn_params['batch_size_cl'], shuffle=True, drop_last=True)#, num_workers=2)
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_pre_trained.parameters(), lr=nn_params['learning_rate'])

    model_trained, optimizer, val_loss, final_epoch = train_cl(
            nn_params['total_epoch_cl'], criterion, model_pre_trained, train_loader, valid_loader, optimizer, nn_params['es_epoch_cl'], nn_params['device'])
    
    return model_trained

# Train simple MLP
def train_classifier_(total_epoch, criterion, model, train_loader, valid_loader, optimizer, es, device):

    best_loss = 1e4
    es_count = 0
    
    # Function for oscillation of learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2)
    
    for epoch in range(total_epoch):

        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):

            # Update learning rate
            scheduler.step()

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output_cl = model(data)
            
            ce_loss = criterion(output_cl, target)
            
            loss = ce_loss

            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):

                data = data.to(device)
                target = target.to(device)

                output = model(data)

                ce_loss1 = criterion(output, target)
                val_loss += ce_loss1.item()
        
        # Check validation loss
        if val_loss<best_loss:
            best_loss = val_loss
            best_model = model
            best_optimizer = optimizer
            es_count = 0
        else:
            es_count += 1

        # Check validation loss
        if es_count >= es:
            #print("Early Stopping")
            break

    return best_model, best_optimizer, best_loss, epoch


# Make dataloader, train model
def classifier_training(x_train, labels_train, x_valid, labels_valid, nn_params):

    input_num = x_train.shape[1]

    train_data = [(Tensor(data), torch.tensor(int(target))) for data, target in zip(x_train, labels_train)]
    valid_data = [(Tensor(data), torch.tensor(int(target))) for data, target in zip(x_valid, labels_valid)]

    train_loader = DataLoader(train_data, batch_size=nn_params['batch_size_cl'], shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=len(valid_data), shuffle=False)

    device = nn_params['device']

    model = NetClassify(input_num, nn_params['prefinal_num'])
    model.apply(weight_init)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=nn_params['learning_rate'])

    model_trained, optimizer, val_loss, final_epoch = train_classifier_(
            nn_params['total_epoch_cl'], criterion, model, train_loader, valid_loader, optimizer, nn_params['es_epoch_cl'], device)
    
    return model_trained

# Initialize model weight
def weight_init(m):

    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)

        if m.bias is not None:
            init.normal_(m.bias.data)
