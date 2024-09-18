# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torchvision.transforms import Resize
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import get_root_logger
from ..builder import BACKBONES
from .resnet3d import ResNet3d

from .resnet3d_slowfast import ResNet3dPathway
from .resnet3d_slowonly import ResNet3dSlowOnly
from .resnet3d_slowonly_local import ResNet3dSlowOnlyLocal
from .resnet3d import ResNet3d
from .resnet3d_local import ResNet3dLocal

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False

pathway_cfg = {
    'resnet3d': ResNet3dPathway,
    'resnet3d_slowonly': ResNet3dSlowOnly,
    'resnet3d_slowonly_local': ResNet3dSlowOnlyLocal,
    'ResNet3d': ResNet3d,
    'ResNet3dLocal': ResNet3dLocal,
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
class ResNet3dSlowOnly2sLFAtt(nn.Module):
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










    def forward(self,full, local, all_crop_boxes):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        # self.local_attention(local, full, all_crop_boxes)
        # full=self.full_path(full)
        # local=self.local_path(local)

        full = self.full_path.conv1(full)
        local = self.local_path.conv1(local)
        if self.full_path.with_pool1:
            full = self.full_path.maxpool(full)
        if self.local_path.with_pool1:
            local = self.local_path.maxpool(local)


        full_layer_names  = self.full_path.res_layers
        local_layer_names = self.local_path.res_layers




        local_res_layer = getattr(self.local_path, local_layer_names[0])
        local = local_res_layer(local)
        if self.local_path.with_pool2:
            local = self.local_path.pool2(local)
        local_res_layer = getattr(self.local_path, local_layer_names[1])
        local = local_res_layer(local)

        full = self.local_attention(local, full, all_crop_boxes)

        full_res_layer = getattr(self.full_path, full_layer_names[0])
        full = full_res_layer(full)
        if self.full_path.with_pool2:
            full = self.full_path.pool2(full)
        full_res_layer = getattr(self.full_path, full_layer_names[1])
        full = full_res_layer(full)
        full_res_layer = getattr(self.full_path, full_layer_names[2])
        full = full_res_layer(full)
        full_res_layer = getattr(self.full_path, full_layer_names[3])
        full = full_res_layer(full)
        local_res_layer = getattr(self.local_path, local_layer_names[2])
        local = local_res_layer(local)
        local_res_layer = getattr(self.local_path, local_layer_names[3])
        local = local_res_layer(local)

        out = (full, local)

        return out

    def local_attention(self, local, full, all_crop_boxes):

        # 1. mean local clips
        clip_mean_size = int(local.shape[1] / self.local_path.groups)           # 512 / 8 = 64
        local_ = []
        for i in range(self.local_path.groups):
            local_.append(torch.mean(local[:, i*clip_mean_size:(i+1)*clip_mean_size, :, :, :], 1))              # [4, 8, 16, 16]
        local_ = torch.stack(([local_[i] for i in range(self.local_path.groups)]), dim=1)                     # [4, (8), 8, 16, 16] [batch_size, clip_size, frame_size, h, w]

        # 2. recover frame
        recover_size = int(full.shape[2] / local.shape[2])
        local_2 = []
        for i in range(local_.size()[2]):
            for _ in range(recover_size):
                local_2.append(local_[:, :, i, :, :])  # [4, 8, 16, 16]                                         # [batchsize, clip_size, h, w]
        local_2 = torch.stack(([local_2[i] for i in range(len(local_2))]), dim=2)                               # local_2 size [4, 8, (16), 16, 16]
                                                                                                                # full.size()  [4, 128, 16, 32, 32]
                                                                                                                #  [batch_size, clip_size, (frame_size), h, w]
        # 3. bulid the mask 用到all crop boxes逐像素生成map
        # 3.1 culculate the from size and to size
        crop_box_norm_size = []
        for batchsize in all_crop_boxes:
            for boxes in batchsize:
                for box in boxes:
                    # print(box)
                    crop_box_norm_size.append([float((box[3] - box[1])), float((box[2] - box[0]))])
                    break
                break
        from_size = [local.size()[3],local.size()[4]]
        to_size = [(int(x_len*full.size()[3]), int(y_len*full.size()[3])) for x_len, y_len in crop_box_norm_size]
        # 3.2 small/resize the local clips [4 batch_size, 8 clip_size, 16 frame_size, 16 (from_size1), 16(from_size2)] -> [8, 16, to_sizeh, to_sizew]*batchsize
        local_3 = []
        for i, batch in enumerate(local_2):
            local_3.append(Resize(size=to_size[i])(local_2[i]))                                                 # [8, 16, to_sizeh, to_sizew]*batchsize

        # 3.3 merge the local attention clips to the mask
        # 创建tensor 一个是mask 一个是用于统计mask的像素点是否重复叠加。
        mask = np.zeros((full.size()[0],full.size()[2],full.size()[3],full.size()[4],))               # mask shape: batchsize, frame_size, h, w
        # all_crop_boxes中的四维batchsize、framesize、clipsize和boxes，其中framesize需要计算膨胀系数，因为local和global的frame channel在 特征提取的过程中进行了压缩。
        # 所以frame_channel膨胀系数在local-full和origin之间都有。
        # 不用管local-full的膨胀系数，前面操作2 recover frame的时候处理过了，只需要处理boxes的framesize和features之间的膨胀系数关系。
        all_crop_boxes = all_crop_boxes.cpu().numpy()
        frame_times = int(all_crop_boxes.shape[1] / full.shape[2])
        for batchsize_level, batchsize_level_feature in enumerate(local_3):
            for clips_level, clips_level_feature in enumerate(batchsize_level_feature):
                for framesize_level, framesize_level_feature in enumerate(clips_level_feature):

                    location = (all_crop_boxes[batchsize_level][framesize_level][clips_level]*full.size()[3]).astype(int)
                    location[3] = location[1] + framesize_level_feature.shape[0]
                    location[2] = location[0] + framesize_level_feature.shape[1]

                    if location[0]>=full.size()[3] or location[1]>=full.size()[3] or location[2]<0 or location[3]<0:
                        continue
                    # if location[0] < 0:
                    #     location[0] = 0
                    # if location[1] < 0:
                    #     location[1] = 0
                    # if location[2] > full.size()[3]:
                    #     location[2] = full.size()[3]
                    # if location[3] > full.size()[4]:
                    #     location[3] = full.size()[4]



                    try:
                        # 先判断Y是不是在范围内，不在跳过，再判断x的范围，然后粘贴到适合的位置就好了
                        # x两个点的位置一共有三的平方有九种可能，排除掉上面去掉的两种以及不可能存在的三种可能，还有剩下四种可能，需要分别处理。
                        for y_number, y in enumerate(range(location[1],location[3])):
                            if y < 0 or y >= full.size()[4]:
                                continue
                            if location[0] < 0:
                                if location[2] <= full.size()[3]:
                                    mask[batchsize_level][framesize_level][y][0:location[2]] = framesize_level_feature.cpu().detach().numpy()[y_number][abs(location[0]):abs(location[0])+location[2]]
                                else:
                                    # location[2] > full.size()[3]:
                                    mask[batchsize_level][framesize_level][y][:] = framesize_level_feature.cpu().detach().numpy()[y_number][abs(location[0]):abs(location[0])+full.size()[3]]
                            if location[0] >= 0:
                                if location[2] <= full.size()[3]:
                                    mask[batchsize_level][framesize_level][y][location[0]:location[2]] = framesize_level_feature.cpu().detach().numpy()[y_number][:]
                                else:
                                    # location[2] > full.size()[3]:
                                    mask[batchsize_level][framesize_level][y][location[0]:] = framesize_level_feature.cpu().detach().numpy()[y_number][:full.size()[3]-location[0]]



                    except:
                        print(mask.shape)
                        print(mask[batchsize_level][framesize_level][y][location[0]:location[2]].shape)
                        print(framesize_level_feature.cpu().detach().numpy().shape)
                        print(y,y_number,location)

                # Image.fromarray(
                #     np.uint8(framesize_level_feature.cpu().detach().numpy() * 255 / 2 )).save(
                #     'full_frame{}.jpg'.format(framesize_level), quality=95,
                #     subsampling=0)
                #
                # Image.fromarray(
                #     np.uint8(mask[batchsize_level][framesize_level] * 255 / 2 )).save(
                #     'full_frame{}_mask.jpg'.format(framesize_level), quality=95,
                #         subsampling=0)
                # Image.fromarray(
                #             np.uint8(full[batchsize_level][0][framesize_level].cpu().detach().numpy() * 255 / 2 )).save(
                #             'full_frame.jpg'.format(framesize_level), quality=95,
                #             subsampling=0)

        mask = torch.from_numpy(mask).cuda() + torch.ones(mask.shape).cuda()
        # mask = torch.from_numpy(mask) + torch.ones(mask.shape)
        # # 打印一下看看mask的内容是否像我想的那样
        # for batchsize_level, batchsize_level_feature in enumerate(full):
        #     for input_channel_level, input_channel_level_feature in enumerate(batchsize_level_feature):
        #         for framesize_level, framesize_level_feature in enumerate(input_channel_level_feature):
        #             # print(framesize_level_feature.shape)
        #
        #             Image.fromarray(
        #                 np.uint8(framesize_level_feature.cpu().detach().numpy() * 255 / 2 )).save(
        #                 'full_frame{}.jpg'.format(framesize_level), quality=95,
        #                 subsampling=0)
        #
        #             Image.fromarray(
        #                 np.uint8(mask[batchsize_level][framesize_level].cpu().detach().numpy() * 255 / 2 )).save(
        #                 'full_frame{}_mask.jpg'.format(framesize_level), quality=95,
        #                 subsampling=0)
        #     break
        mask = mask.unsqueeze(1)
        full = torch.mul(full, mask)
        full = full.type_as(local)
        return full


if mmdet_imported:
    MMDET_BACKBONES.register_module()(ResNet3dSlowOnly2sLFAtt)
