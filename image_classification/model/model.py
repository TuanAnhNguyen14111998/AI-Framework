import torch.nn as nn
import math
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self, model_name="efficientnet-b0", n_class=29):
        super(Net, self).__init__()
        self.model_name = model_name
        self.backbone = EfficientNet.from_pretrained(model_name, num_classes=n_class)
        
    def forward(self, x):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        output = self.backbone(x)

        return output
