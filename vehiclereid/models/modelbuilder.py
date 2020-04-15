import torch.nn as nn
from .reid import PCBModel

class ModelBuilder(nn.Module):
    def __init__(self, model, num_classes, loss):
        super(ModelBuilder, self).__init__()
        self.loss = loss
        self.model = model
        self.reid_model = PCBModel(num_classes=num_classes, loss = loss)

    def forward(self, x):
        x = self.model(x)
        x = self.reid_model(x)
        return x
