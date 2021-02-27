import torch.nn as nn

def get_loss(outputs, labels):
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    return loss
