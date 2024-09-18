# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class EightFCHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc1 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc2 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc3 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc4 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc5 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc6 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc7 = nn.Linear(int(self.in_channels / 8), self.num_classes)
        self.fc8 = nn.Linear(int(self.in_channels / 8), self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)
        normal_init(self.fc2, std=self.init_std)
        normal_init(self.fc3, std=self.init_std)
        normal_init(self.fc4, std=self.init_std)
        normal_init(self.fc5, std=self.init_std)
        normal_init(self.fc6, std=self.init_std)
        normal_init(self.fc7, std=self.init_std)
        normal_init(self.fc8, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), 8, int(x.size(1) / 8))
        if self.dropout is not None:
            x[:, 0] = self.dropout(x[:, 0])
            x[:, 1] = self.dropout(x[:, 1])
            x[:, 2] = self.dropout(x[:, 2])
            x[:, 3] = self.dropout(x[:, 3])
            x[:, 4] = self.dropout(x[:, 4])
            x[:, 5] = self.dropout(x[:, 5])
            x[:, 6] = self.dropout(x[:, 6])
            x[:, 7] = self.dropout(x[:, 7])

        x1 = self.fc1(x[:, 0])
        x2 = self.fc2(x[:, 1])
        x3 = self.fc3(x[:, 2])
        x4 = self.fc4(x[:, 3])
        x5 = self.fc5(x[:, 4])
        x6 = self.fc6(x[:, 5])
        x7 = self.fc7(x[:, 6])
        x8 = self.fc8(x[:, 7])
        cls_score = torch.mean(torch.stack((x1, x2, x3, x4, x5, x6, x7, x8)), dim=0)
        # [N, num_classes]
        return cls_score
