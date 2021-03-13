import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch

def get_loss(outputs, labels):
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    return loss
