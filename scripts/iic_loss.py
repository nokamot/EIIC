import sys
import math
import torch

# Calculate a matrix of joint probability distribution
def compute_joint(x_out, x_tf_out):

    
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
    
    p_i_j = p_i_j.sum(dim=0)
    
    p_i_j = (p_i_j + p_i_j.t()) / 2.

    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j
    
def IID_loss(x_out, x_tf_out, alpha, EPS=sys.float_info.epsilon):
    
    bs, k = x_out.size()
    # Joint distribution
    p_i_j = compute_joint(x_out, x_tf_out)  

    # Marginal distribution
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor(
        [EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    # Introduce alpha into mutual information loss for weighting more on marginal entropy terms
    loss = -1*(p_i_j * (torch.log(p_i_j) - alpha *
                        torch.log(p_j) - alpha*torch.log(p_i))).sum()/math.log(k)

    return loss
