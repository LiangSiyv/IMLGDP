# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class Four2OneFCHead(BaseHead):
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
                 alphas=[1.0, 1.0, 1.0, 1.0],
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc1 = nn.Linear(int(self.in_channels / 4), self.num_classes)
        self.alphas=alphas

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc1, std=self.init_std)

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
        x = x.view(x.size(0), 4, int(x.size(1) / 4))

        # 头，头，左肩，右肩，左肘，右肘，左手，右手八个点的权重是[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if self.dropout is not None:
            x[:, 0] = self.dropout(x[:, 0])
            x[:, 1] = self.dropout(x[:, 1])
            x[:, 2] = self.dropout(x[:, 2])
            x[:, 3] = self.dropout(x[:, 3])

        alphas=torch.tensor(self.alphas, device=x.device)
        x1 = self.fc1(x[:, 0]) * alphas[0]
        x2 = self.fc1(x[:, 1]) * alphas[1]
        x3 = self.fc1(x[:, 2]) * alphas[2]
        x4 = self.fc1(x[:, 3]) * alphas[3]
        cls_score = torch.mean(torch.stack((x1, x2, x3, x4)), dim=0)
        # [N, num_classes]
        return cls_score
