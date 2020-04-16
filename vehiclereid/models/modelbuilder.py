import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from .reid import PCBModel, BinPCBModel

class ModelBuilder(nn.Module):
    def __init__(self, model, num_classes, loss):
        super(ModelBuilder, self).__init__()
        self.loss = loss
        self.model_main = model
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        if num_classes >0:
            self.model_weight = model
            self.reid_model = BinPCBModel(num_classes=num_classes, loss = loss)
            self.linlayer = nn.Linear(2048, num_classes)

    def forward(self, x):
        x_main = self.model_main(x)
        if self.training:
            x_weight = self.model_weight(x)
            x_weight = self.reid_model(x_weight)
            if self.training and 'htri' in self.loss:
                flat_feat, cls_feat, x = x_weight
            elif self.training and 'xent' in self.loss:
                cls_feat, x = x_weight
            weight,_ = torch.max(x, 1)
            if self.training and 'htri' in self.loss:
                main_feat, _, _ = x_main
            elif self.training and 'xent' in self.loss:
                main_feat, _ = x_main
            main_feat = main_feat[-1]
            main_feat = weight.unsqueeze(1) * main_feat
            gap_feat = self.global_avgpool(main_feat)
            mflat_feat = gap_feat.view(gap_feat.size(0), -1)
            mcls_feat = self.linlayer(mflat_feat)
            local_feat = x.view(x.size(0), x.size(1), -1)
            local_feat_list = [local_feat[:,:,i].squeeze() for i in range(local_feat.size(-1))]
            return local_feat_list, mflat_feat, mcls_feat

        mflat_feat = x_main[-1]
        return mflat_feat




