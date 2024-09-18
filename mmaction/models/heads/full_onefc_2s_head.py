# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class FullOneFC2sHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 alpha=1.0,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std


        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.fc1 = nn.Linear(int(self.in_channels / 8), self.num_classes)          # local and full
        # self.fc1 = nn.Linear(int(self.in_channels / 2), self.num_classes)              # large_local and full
        self.alpha = alpha

        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        normal_init(self.fc1, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        x_full, x_local = x
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        x_full = self.avg_pool(x_full)              # shape=[1,2048,32,7,7] -> shape=[1,2048,1,1,1]
        x_local = self.avg_pool(x_local)            # shape=[1,2048,32,7,7] -> shape=[1,2048,1,1,1]
        # [N, channel_fast + channel_slow, 1, 1, 1]
        x_local = x_local.view(x_local.size(0), 8, int(x_local.size(1) / 8))    # shape=[1,2048,1,1,1] -> shape=[1,8,256]





        if self.dropout is not None:
            x_full = self.dropout(x_full)
            x_local[:, 0] = self.dropout(x_local[:, 0])
            x_local[:, 1] = self.dropout(x_local[:, 1])
            x_local[:, 2] = self.dropout(x_local[:, 2])
            x_local[:, 3] = self.dropout(x_local[:, 3])
            x_local[:, 4] = self.dropout(x_local[:, 4])
            x_local[:, 5] = self.dropout(x_local[:, 5])
            x_local[:, 6] = self.dropout(x_local[:, 6])
            x_local[:, 7] = self.dropout(x_local[:, 7])

        # [N x C]
        x_full = x_full.view(x_full.size(0), -1)    # shape=[1,2048,1,1,1] -> shape=[1,2048]


        x1 = self.fc1(x_local[:, 0])                # shape=[1,1067]
        x2 = self.fc1(x_local[:, 1])                # shape=[1,1067]
        x3 = self.fc1(x_local[:, 2])                # shape=[1,1067]
        x4 = self.fc1(x_local[:, 3])                # shape=[1,1067]
        x5 = self.fc1(x_local[:, 4])                # shape=[1,1067]
        x6 = self.fc1(x_local[:, 5])                # shape=[1,1067]
        x7 = self.fc1(x_local[:, 6])                # shape=[1,1067]
        x8 = self.fc1(x_local[:, 7])                # shape=[1,1067]
        cls_score_local = torch.mean(torch.stack((x1, x2, x3, x4, x5, x6, x7, x8)), dim=0)  # shape=[1,1067]

        # # [N x num_classes]
        # cls_score_full = self.fc_cls(x_full)        # shape=[1,1067]
        # # [N, channel_fast + channel_slow, 1, 1, 1]
        # x_local = x_local.view(x_local.size(0), 8, int(x_local.size(1) / 8))    # shape=[1,2048,1,1,1] -> shape=[1,8,256]





        if self.dropout is not None:
            x_full = self.dropout(x_full)
            x_local[:, 0] = self.dropout(x_local[:, 0])
            x_local[:, 1] = self.dropout(x_local[:, 1])
            x_local[:, 2] = self.dropout(x_local[:, 2])
            x_local[:, 3] = self.dropout(x_local[:, 3])
            x_local[:, 4] = self.dropout(x_local[:, 4])
            x_local[:, 5] = self.dropout(x_local[:, 5])
            x_local[:, 6] = self.dropout(x_local[:, 6])
            x_local[:, 7] = self.dropout(x_local[:, 7])

        # [N x C]
        x_full = x_full.view(x_full.size(0), -1)    # shape=[1,2048,1,1,1] -> shape=[1,2048]


        x1 = self.fc1(x_local[:, 0])                # shape=[1,1067]
        x2 = self.fc1(x_local[:, 1])                # shape=[1,1067]
        x3 = self.fc1(x_local[:, 2])                # shape=[1,1067]
        x4 = self.fc1(x_local[:, 3])                # shape=[1,1067]
        x5 = self.fc1(x_local[:, 4])                # shape=[1,1067]
        x6 = self.fc1(x_local[:, 5])                # shape=[1,1067]
        x7 = self.fc1(x_local[:, 6])                # shape=[1,1067]
        x8 = self.fc1(x_local[:, 7])                # shape=[1,1067]
        cls_score_local = torch.mean(torch.stack((x1, x2, x3, x4, x5, x6, x7, x8)), dim=0)  # shape=[1,1067]

        # [N x num_classes]
        cls_score_full = self.fc_cls(x_full)        # shape=[1,1067]
        # print( 'cls_score_full.device',cls_score_full.device,'cls_score_local.device',cls_score_local.device,'self.aplha.device',self.alpha.device)
        alpha=torch.tensor([self.alpha], device=cls_score_local.device)
        return cls_score_full + cls_score_local * alpha
