import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from siammask.experiments.siammask_sharp.resnet import resnet50
import pdb


def resnet(layer3=True, layer4=True):
    return resnet50(layer3=layer3, layer4=layer4)

class Block(nn.Module):
    def __init__(self, blocks=[3,4]):
        super(Block, self).__init__()
        self.blocks = blocks
        if len(blocks) >1:
            self.channel_down = ChannelDown(1024, 512)
        for item in blocks:
            self.add_module('layer'+str(item), getattr(resnet(), 'layer'+str(item)))

    def forward(self, x):
        if len(self.blocks)>1:
            x = self.channel_down(x)
        for item in self.blocks:
            x = eval('self.layer' + str(item))(x)
        return x

class ChannelDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelDown, self).__init__()
        self.channel_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.channel_down(x)
        return x

class PCBModel(nn.Module):
  def __init__(
      self,
      dropout=None,
      anchors=None,
      loss = {},
      blocks=[],
      linear_layer_cfg=[],
      linear_relu_cfg=[],
      linear_bn_cfg=[],
      linear_do_cfg=[],
      last_conv_stride=1,
      last_conv_dilation=1,
      num_stripes=6,
      num_features=256,
      num_classes=9):
    super(PCBModel, self).__init__()

    self.num_stripes = num_stripes
    self.blocks = blocks
    self.loss = loss
    self.block_modules = Block(blocks)
    self.local_conv_list = nn.ModuleList()
    for _ in range(num_stripes):
      self.local_conv_list.append(nn.Sequential(
        nn.Conv2d(512, num_features, 1),
        nn.BatchNorm2d(num_features),
        nn.ReLU(inplace=True)
      ))


    if linear_layer_cfg:
        _linear_layers = [num_features*num_stripes] + linear_layer_cfg
        all_ll = []
        for i in range(len(linear_layer_cfg)):
            fc = nn.Linear(_linear_layers[i], _linear_layers[i+1])
            all_ll.append(fc)
            if linear_relu_cfg[i]:
                all_ll.append(nn.ReLU(inplace=True))
            if linear_bn_cfg[i]:
                all_ll.append(nn.BatchNorm1d(linear_layer_cfg[i]))
            if linear_do_cfg[i]:
                all_ll.append(nn.Dropout(0.5))

        self.globalfc_list = nn.Sequential(*all_ll)

        for item in self.globalfc_list.modules():
            if isinstance(item, nn.Linear):
                init.normal_(item.weight, std=0.001)
                init.constant_(item.bias, 0)

        if num_classes > 0:
            self.finalfc = nn.Linear(linear_layer_cfg[-1], num_classes)
            init.normal_(self.finalfc.weight, std=0.001)
            init.constant_(self.finalfc.bias, 0)

    if num_classes > 0:
      self.fc_list = nn.ModuleList()
      for _ in range(num_stripes):
        fc = nn.Linear(num_features, num_classes)
        init.normal_(fc.weight, std=0.001)
        init.constant_(fc.bias, 0)
        self.fc_list.append(fc)

  def forward(self, x):
    """
    Returns:
      local_feat_list: each member with shape [N, c]
    """
    # shape [N, C, H, W]
    if ('htri' in self.loss) and self.training:
        all_feat, cls_feat, flat_feat = x
    elif ('xent' in self.loss) and self.training:
        all_feat, cls_feat = x
    else:
        all_feat, flat_feat = x

    x = all_feat[1]
    if self.blocks:
        x = self.block_modules(x)

    x = x.chunk(6, 3)
    local_feat_list = []
    logits_list = []
    for i in range(self.num_stripes):
      # shape [N, C, 1, 1]
      kh,kw = x[i].shape[2:]
      local_feat = F.avg_pool2d(
        x[i], (kh, kw))
      # shape [N, c, 1, 1]
      local_feat = self.local_conv_list[i](local_feat)
      # shape [N, c]
      local_feat = local_feat.view(local_feat.size(0), -1)
      local_feat_list.append(local_feat)
      if hasattr(self, 'fc_list'):
        logits_list.append(self.fc_list[i](local_feat))

    global_feat = torch.cat(local_feat_list, 1)

    if hasattr(self, 'globalfc_list'):
        global_feat = self.globalfc_list(global_feat)

    if hasattr(self, 'finalfc') and self.training:
        global_logits = self.finalfc(global_feat)
        return local_feat_list, logits_list, global_logits


    if hasattr(self, 'fc_list') and self.training:
        return local_feat_list, logits_list, cls_feat, flat_feat

    if not self.training:
        return local_feat_list, global_feat, flat_feat

class BinPCBModel(nn.Module):
    def __init__(self, loss, in_channels=2048,
                 num_classes=575,
                 layer_cfg=[1024],
                 relu_cfg=[1],
                 bn_cfg=[1],
                 do_cfg=[1],
                 bias_cfg=[1]
                ):
        super(BinPCBModel, self).__init__()
        self.loss = loss
        _channels_cfg = [in_channels] + layer_cfg + [num_classes]
        all_layers = []
        if layer_cfg:
            for i in range(len(layer_cfg)):
                all_layers.append(nn.Conv2d(_channels_cfg[i], _channels_cfg[i+1], 1, bias=True if bias_cfg[i] else False))
                if relu_cfg[i]:
                    all_layers.append(nn.ReLU(inplace=True))
                if bn_cfg[i]:
                    all_layers.append(nn.BatchNorm2d(_channels_cfg[i+1]))
                if do_cfg[i]:
                    all_layers.append(nn.Dropout2d())
            all_layers.append(nn.Conv2d(_channels_cfg[-2], _channels_cfg[-1], 1))
        else:
            all_layers.append(nn.Conv2d(in_channels, num_classes, 1))
        self.bc = nn.Sequential(*all_layers)

    def forward(self, x):
        # shape [N, C, H, W]
        if ('htri' in self.loss) and self.training:
            all_feat, cls_feat, flat_feat = x
        elif ('xent' in self.loss) and self.training:
            all_feat, cls_feat = x
        else:
            all_feat, flat_feat = x

        x = all_feat[-1]
        x = self.bc(x)

        if self.training and 'htri' in self.loss:
            return flat_feat, cls_feat, x
        if self.training and 'xent' in self.loss:
            return cls_feat, x

        return flat_feat


if __name__ == '__main__':
    model = PCBModel()
    x = torch.rand(64,256,31,31)
    out = model(x)
