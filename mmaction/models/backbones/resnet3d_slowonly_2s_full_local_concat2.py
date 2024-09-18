
# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d import ResNet3d

from .resnet3d_slowfast import ResNet3dPathway
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet3d_slowonly_local import ResNet3dSlowOnlyLocal

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    'resnet3d_slowonly': ResNet3dSlowOnly,
    'resnet3d_slowonly_local': ResNet3dSlowOnlyLocal,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg, *args, **kwargs):
    """Build pathway.

    Args:
        cfg (None or dict): cfg should contain:
            - type (str): identify conv layer type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and 'type' in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop('type')
    if pathway_type not in pathway_cfg:
        raise KeyError(f'Unrecognized pathway type {pathway_type}')

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@BACKBONES.register_module()
class ResNet3dSlowOnly2sFLC2(nn.Module):
    """full local 2s backbone.

    下面需要修改
    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames. The actual resample rate is calculated by
            multipling the ``interval`` in ``SampleFrames`` in the
            pipeline with ``resample_rate``, equivalent to the :math:`\\tau`
            in the paper, i.e. it processes only one out of
            ``resample_rate * interval`` frames. Default: 8.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        slow_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        fast_pathway (dict): Configuration of fast branch, similar to
            `slow_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(self,
                 pretrained,
                 full_pathway,
                 local_pathway):
        self.pretrained = pretrained
        super().__init__()

        self.full_path = build_pathway(full_pathway)
        self.local_path = build_pathway(local_pathway)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(256, 1024)


    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.local_path.init_weights()
            self.full_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x, x_local):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x_full = self.full_path(x)
        x_local = self.local_path(x_local)
        x_full = self.avg_pool(x_full)              # shape=[1,2048,32,7,7] -> shape=[1,2048,1,1,1]
        x_local = self.avg_pool(x_local)            # shape=[1,2048,32,7,7] -> shape=[1,2048,1,1,1]

        # [N, channel_fast + channel_slow, 1, 1, 1]
        x_local = x_local.view(x_local.size(0), 8, int(x_local.size(1) / 8))    # shape=[1,2048,1,1,1] -> shape=[1,8,256]


        # [N x C]
        x_full = x_full.view(x_full.size(0), -1)    # shape=[1,2048,1,1,1] -> shape=[1,2048]


        x1 = self.fc1(x_local[:, 0])                # shape=[1,1024]
        x2 = self.fc1(x_local[:, 1])                # shape=[1,1024]
        x3 = self.fc1(x_local[:, 2])                # shape=[1,1024]
        x4 = self.fc1(x_local[:, 3])                # shape=[1,1024]
        x5 = self.fc1(x_local[:, 4])                # shape=[1,1024]
        x6 = self.fc1(x_local[:, 5])                # shape=[1,1024]
        x7 = self.fc1(x_local[:, 6])                # shape=[1,1024]
        x8 = self.fc1(x_local[:, 7])                # shape=[1,1024]
        cls_score_local = torch.mean(torch.stack((x1, x2, x3, x4, x5, x6, x7, x8)), dim=0)  # shape=[1,1024]

        # [N x num_classes]
        cls_score_full = self.fc_cls(x_full)        # shape=[1,1024]
        out = torch.cat((cls_score_full, cls_score_local),dim=1)

        return out


if mmdet_imported:
    MMDET_BACKBONES.register_module()(ResNet3dSlowOnly2sFLC2)
