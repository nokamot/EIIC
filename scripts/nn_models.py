import torch
import torch.nn as nn
import torch.nn.functional as F

# EIIC model
class NetIIC(nn.Module):
    def __init__(self, input_num, prefinal_num, oc_class_num):
        super(NetIIC, self).__init__()
        
        self.dense00 = nn.Linear(input_num, 3000)
        self.bn00 = nn.BatchNorm1d(3000)
        
        self.dense0 = nn.Linear(3000,1000)
        self.bn0 = nn.BatchNorm1d(1000)
        
        self.dense05 = nn.Linear(1000,500)
        self.bn05 = nn.BatchNorm1d(500)
        
        self.dense1 = nn.Linear(500,prefinal_num)
        self.bn1 = nn.BatchNorm1d(prefinal_num)
        
        self.dense3 = nn.Linear(prefinal_num, 2)
        self.dense3_oc = nn.Linear(prefinal_num, oc_class_num)


        self.dense3_cl = nn.Linear(prefinal_num, 2)
        
    def forward(self, x):
        x = F.relu(self.bn00(self.dense00(x)))
        x = F.relu(self.bn0(self.dense0(x)))
        x = F.relu(self.bn05(self.dense05(x)))
        x_prefinal = F.relu(self.bn1(self.dense1(x)))
        
        y = F.softmax(self.dense3(x_prefinal), dim=1)
        y_oc = F.softmax(self.dense3_oc(x_prefinal), dim=1)
        y_rev_ = 1-y
        
        y_cl = F.softmax(self.dense3_cl(x_prefinal), dim=1)
        
        return y, y_oc, y_rev_, y_cl

# Simple MLP without contrastive learning
class NetClassify(nn.Module):
    def __init__(self, input_num, prefinal_num):
        super(NetClassify, self).__init__()
        
        self.dense00 = nn.Linear(input_num, 3000)
        self.bn00 = nn.BatchNorm1d(3000)
        
        self.dense0 = nn.Linear(3000,1000)
        self.bn0 = nn.BatchNorm1d(1000)
        
        self.dense05 = nn.Linear(1000,500)
        self.bn05 = nn.BatchNorm1d(500)
        
        self.dense1 = nn.Linear(500,prefinal_num)
        self.bn1 = nn.BatchNorm1d(prefinal_num)
        
        self.dense3_cl = nn.Linear(prefinal_num, 2)
        
        
    def forward(self, x):
        x = F.relu(self.bn00(self.dense00(x)))
        x = F.relu(self.bn0(self.dense0(x)))
        x = F.relu(self.bn05(self.dense05(x)))
        x_prefinal = F.relu(self.bn1(self.dense1(x)))
        
        y_cl = F.softmax(self.dense3_cl(x_prefinal), dim=1)
        
        return y_cl
