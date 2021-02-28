import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch

def get_loss(outputs, labels):
    num_examples = outputs.size()[0]
    
    return -torch.sum(outputs[range(num_examples), labels])/num_examples
