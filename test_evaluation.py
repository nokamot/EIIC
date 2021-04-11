import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

def test(model, device, train_loader):
    model.eval()

    out_targs = []
    ref_targs = []
    cnt = 0

    with torch.no_grad():
        for data, target in train_loader:
            cnt += 1
            data = data.to(device)
            target = target.to(device)
            _, _, _, outputs = model(data)

            out_targs.append(outputs.cpu())
            ref_targs.append(target.cpu())

    out_targs = torch.cat(out_targs)
    ref_targs = torch.cat(ref_targs)

    return out_targs.numpy(), ref_targs.numpy()

# Calculate confusion matrix from trained model
def matrix_from_models(models, scalers, x_test, labels_test, num_classes, device):
    
    pred = np.zeros(shape=[x_test.shape[0], num_classes])
    
    for model_trained, scaler in zip(models, scalers):
        
        model_trained.to(device)

        x_test = scaler.transform(x_test)
        test_data = [(Tensor(data), torch.tensor(target)) for data, target in zip(x_test, labels_test)]
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)#, num_workers=2)
        out_targs, ref_targs = test(model_trained, device, test_loader)
        pred += out_targs
        
    # Calculate the mean prediction value of each model's output
    pred /= len(models)
    out_targs = np.argmax(pred, axis=1)
    
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(out_targs)):
        row = ref_targs[i]
        col = out_targs[i]
        matrix[row][col] += 1
        
    return matrix

def test_cl(model, device, train_loader):
    model.eval()

    out_targs = []
    ref_targs = []
    cnt = 0

    with torch.no_grad():
        for data, target in train_loader:
            cnt += 1
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)

            out_targs.append(outputs.cpu())
            ref_targs.append(target.cpu())

    out_targs = torch.cat(out_targs)
    ref_targs = torch.cat(ref_targs)

    return out_targs.numpy(), ref_targs.numpy()

# Calculate confusion matrix from trained model
def matrix_from_models_cl(models, scalers, x_test, labels_test, num_classes, device):
    
    pred = np.zeros(shape=[x_test.shape[0], num_classes])
    
    for model_trained, scaler in zip(models, scalers):
        
        model_trained.to(device)

        x_test = scaler.transform(x_test)
        test_data = [(Tensor(data), torch.tensor(target)) for data, target in zip(x_test, labels_test)]
        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)#, num_workers=2)
        out_targs, ref_targs = test_cl(model_trained, device, test_loader)
        pred += out_targs
        
    # Calculate the mean prediction value of each model's output
    pred /= len(models)
    out_targs = np.argmax(pred, axis=1)
    
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(out_targs)):
        row = ref_targs[i]
        col = out_targs[i]
        matrix[row][col] += 1
        
    return matrix
