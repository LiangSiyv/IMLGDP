# Copyright (c) OpenMMLab. All rights reserved.
from .agcn import AGCN
from .c3d import C3D
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v2_tsm import MobileNetV2TSM
from .resnet import ResNet
from .resnet2plus1d import ResNet2Plus1d
from .resnet3d import ResNet3d, ResNet3dLayer
from .resnet3d_csn import ResNet3dCSN
from .resnet3d_slowfast import ResNet3dSlowFast
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet_audio import ResNetAudio
from .resnet_tin import ResNetTIN
from .resnet_tsm import ResNetTSM
from .stgcn import STGCN
from .tanet import TANet
from .timesformer import TimeSformer
from .x3d import X3D
from .swin_transformer import SwinTransformer3D
from .convnext import ConvNeXt
from .resnet3d_slowonly_local import ResNet3dSlowOnlyLocal
from .resnet3d_slowonly_2s_full_local import ResNet3dSlowOnly2sFL
from .resnet3d_slowonly_2s_lfatt import ResNet3dSlowOnly2sLFAtt
from .resnet3d_slowonly_2s_full_local_concat1 import ResNet3dSlowOnly2sFLC1
from .resnet3d_slowonly_2s_full_local_concat2 import ResNet3dSlowOnly2sFLC2
from .resnet3d_slowonly_2s_full_local_concat3 import ResNet3dSlowOnly2sFLC3

__all__ = [
    'C3D', 'ResNet', 'ResNet3d', 'ResNetTSM', 'ResNet2Plus1d',
    'ResNet3dSlowFast', 'ResNet3dSlowOnly', 'ResNet3dCSN', 'ResNetTIN', 'X3D',
    'ResNetAudio', 'ResNet3dLayer', 'MobileNetV2TSM', 'MobileNetV2', 'TANet',
    'TimeSformer', 'STGCN', 'AGCN', 'SwinTransformer3D', 'ConvNeXt',
    'ResNet3dSlowOnlyLocal', 'ResNet3dSlowOnly2sFL', 'ResNet3dSlowOnly2sLFAtt',
    'ResNet3dSlowOnly2sFLC1', 'ResNet3dSlowOnly2sFLC2', 'ResNet3dSlowOnly2sFLC3'
]
