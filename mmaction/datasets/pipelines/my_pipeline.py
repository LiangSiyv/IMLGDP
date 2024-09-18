import mmcv
import cv2
from ..builder import PIPELINES
import numpy as np
from collections.abc import Sequence
from math import floor
from .augmentations import _init_lazy_if_proper
import random


# RandomSkeletonNoise

@PIPELINES.register_module()
class RandomSkeletonNoise:

    def __init__(self, noise_rate, noisze_method=1):
        self.noise_rate=noise_rate
        self.noisze_method=noisze_method

    def __call__(self, results):
        skeleton = results['skeleton']
        frame_width = results['original_shape'][1]
        frame_height = results['original_shape'][0]

        for i,frame_ind in enumerate(results['frame_inds']):
            flag = frame_ind >= 150
            while flag:
                frame_ind = frame_ind - 150
                flag = frame_ind >= 150
            for j in range(skeleton.shape[1]):
                noise_rate_x =  random.uniform(-self.noise_rate, self.noise_rate)
                noise_rate_y = random.uniform(-self.noise_rate, self.noise_rate)
                skeleton[frame_ind, j, 0, 0] = skeleton[frame_ind, j, 0, 0] + noise_rate_x*frame_width
                skeleton[frame_ind, j, 1, 0] = skeleton[frame_ind, j, 1, 0] + noise_rate_y*frame_height
        results['skeleton']=skeleton

        return results



@PIPELINES.register_module()
class LocalRandomRotation:

    def __init__(self, p, angle, center=None, scale=1.0):
        self.p = p
        self.angle = eval(angle) if isinstance(angle, str) else angle
        self.center = center
        self.scale = scale

    def __call__(self, results):
        local_imgs = results['local_imgs']
        if np.random.uniform(0, 1) < self.p:
            min, max = self.angle
            degree = np.random.randint(min, max)
        else:
            degree = 0
        M = cv2.getRotationMatrix2D(self.center, degree, self.scale)

        frame_size, clip_size = np.array(local_imgs).shape[0], np.array(local_imgs).shape[1]
        for i in range(frame_size):
            for j in range(clip_size):
                frame_width = local_imgs[i][j].shape[1]
                frame_height = local_imgs[i][j].shape[0]
                if self.center == None:
                    self.center = (frame_width / 2, frame_height / 2)
                local_imgs[i][j] = cv2.warpAffine(local_imgs[i][j], M, (frame_width, frame_height))


        return results


@PIPELINES.register_module()
class Local_Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, local_imgs, modality):
        for local_img in local_imgs:
            _ = [mmcv.imflip_(img, self.direction) for img in local_img]
        lt = len(local_imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                local_imgs[i] = mmcv.iminvert(local_imgs[i])
        return local_imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            if flip:
                if 'local_imgs' in results:
                    results['local_imgs'] = self._flip_imgs(results['local_imgs'],
                                                      modality)
                if 'keypoint' in results:
                    kp = results['keypoint']
                    kpscore = results.get('keypoint_score', None)
                    kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                    results['keypoint'] = kp
                    if 'keypoint_score' in results:
                        results['keypoint_score'] = kpscore
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            lazyop['flip'] = flip
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            assert not self.lazy and self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str









# augmentations

# 添加一个用于skeleton_guide_8_crop，直接处理imgs成8个×crop_shape的局部图像

@PIPELINES.register_module()
class SkeletonGuide8Crop:

    def __init__(self, box_scale, crop_number=8):
        self.box_scale=box_scale
        self.crop_number=crop_number

    @staticmethod
    def _crop_imgs(imgs, crop_bboxes):
        x1, y1, x2, y2 = crop_bboxes
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _create_crop_box(center_point, image_width, image_height, box_len):
        # make crop_box
        # print(center_point)
        crop_box = [floor(center_point[0] - box_len / 2), floor(center_point[1] - box_len / 2),
                    floor(center_point[0] + box_len / 2), floor(center_point[1] + box_len / 2)]  # 左上xy 右下xy
        # print(crop_box)

        # check crop_box is square




        # check crop_box point in the img
        if crop_box[0] < 0:
            crop_box = [int(i) for i in list(np.array(crop_box) + np.array([abs(crop_box[0]), 0, abs(crop_box[0]), 0]))]
        if crop_box[1] < 0:
            crop_box = [int(i) for i in list(np.array(crop_box) + np.array([0, abs(crop_box[1]), 0, abs(crop_box[1])]))]
        if crop_box[2] > image_width:
            crop_box = [int(i) for i in list(
                np.array(crop_box) - np.array([abs(image_width - crop_box[2]), 0, abs(image_width - crop_box[2]), 0]))]
        if crop_box[3] > image_height:
            crop_box = [int(i) for i in list(np.array(crop_box) - np.array(
                [0, abs(image_height - crop_box[3]), 0, abs(image_height - crop_box[3])]))]
        # print(crop_box, crop_box[2]-crop_box[0], crop_box[3]-crop_box[1])

        assert crop_box[2] - crop_box[0] == box_len
        assert crop_box[3] - crop_box[1] == box_len

        return crop_box

    def __call__(self, results):


        imgs=results['original_imgs']
        skeleton = results['skeleton']
        box_scale = self.box_scale
        frame_width = results['original_shape'][1]
        frame_height = results['original_shape'][0]

        all_crop_boxes = []

        local_imgs = []
        for i,frame_ind in enumerate(results['frame_inds']):
            flag = frame_ind >= 150
            while flag:
                frame_ind = frame_ind - 150
                flag = frame_ind >= 150
            box_len = np.mean(np.linalg.norm(skeleton[:, 0, :2, :] - skeleton[:, 1, :2, :],
                                             axis=1)) * box_scale  # point 0 到point1的平均距离，乘以三可以作为捕捉框的大小
            image_a = []
            crop_boxs = []
            if self.crop_number==8:
                for j in [0, 0, 1, 2, 3, 4, 11, 21]:
                    # parameter set
                    center_point = (int(frame_width - skeleton[frame_ind, j, 0, 0]), int(skeleton[frame_ind, j, 1, 0]))
                    # make crop_box
                    x1,y1,x2,y2 = self._create_crop_box(center_point, frame_width, frame_height, int(box_len))
                    crop_box=[x1,y1,x2,y2]

                    image_a.append(imgs[i][y1:y2, x1:x2])

                    # norm crop_box
                    norm_crop_box = [crop_box[0]/frame_width,crop_box[1]/frame_height,crop_box[2]/frame_width,crop_box[3]/frame_height]
                    crop_boxs.append(norm_crop_box)
            elif self.crop_number==4:
                for j in [0, 0, 11, 21]:
                    # parameter set
                    center_point = (int(frame_width - skeleton[frame_ind, j, 0, 0]), int(skeleton[frame_ind, j, 1, 0]))
                    # make crop_box
                    x1,y1,x2,y2 = self._create_crop_box(center_point, frame_width, frame_height, int(box_len))
                    crop_box=[x1,y1,x2,y2]

                    image_a.append(imgs[i][y1:y2, x1:x2])

                    # norm crop_box
                    norm_crop_box = [crop_box[0]/frame_width,crop_box[1]/frame_height,crop_box[2]/frame_width,crop_box[3]/frame_height]
                    crop_boxs.append(norm_crop_box)
            all_crop_boxes.append(crop_boxs)

            local_imgs.append(image_a)

        results['local_imgs']=local_imgs
        results['all_crop_boxes']=all_crop_boxes

        return results



@PIPELINES.register_module()
class SkeletonGuide4Crop:

    def __init__(self, box_scale):
        self.box_scale=box_scale

    @staticmethod
    def _crop_imgs(imgs, crop_bboxes):
        x1, y1, x2, y2 = crop_bboxes
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _create_crop_box(center_point, image_width, image_height, box_len):
        # make crop_box
        # print(center_point)
        crop_box = [floor(center_point[0] - box_len / 2), floor(center_point[1] - box_len / 2),
                    floor(center_point[0] + box_len / 2), floor(center_point[1] + box_len / 2)]  # 左上xy 右下xy
        # print(crop_box)

        # check crop_box is square




        # check crop_box point in the img
        if crop_box[0] < 0:
            crop_box = [int(i) for i in list(np.array(crop_box) + np.array([abs(crop_box[0]), 0, abs(crop_box[0]), 0]))]
        if crop_box[1] < 0:
            crop_box = [int(i) for i in list(np.array(crop_box) + np.array([0, abs(crop_box[1]), 0, abs(crop_box[1])]))]
        if crop_box[2] > image_width:
            crop_box = [int(i) for i in list(
                np.array(crop_box) - np.array([abs(image_width - crop_box[2]), 0, abs(image_width - crop_box[2]), 0]))]
        if crop_box[3] > image_height:
            crop_box = [int(i) for i in list(np.array(crop_box) - np.array(
                [0, abs(image_height - crop_box[3]), 0, abs(image_height - crop_box[3])]))]
        # print(crop_box, crop_box[2]-crop_box[0], crop_box[3]-crop_box[1])

        assert crop_box[2] - crop_box[0] == box_len
        assert crop_box[3] - crop_box[1] == box_len

        return crop_box

    def __call__(self, results):


        imgs=results['original_imgs']
        skeleton = results['skeleton']
        box_scale = self.box_scale
        frame_width = results['original_shape'][1]
        frame_height = results['original_shape'][0]

        all_crop_boxes = []

        local_imgs = []
        for i,frame_ind in enumerate(results['frame_inds']):
            flag = frame_ind >= 150
            while flag:
                frame_ind = frame_ind - 150
                flag = frame_ind >= 150
            box_len = np.mean(np.linalg.norm(skeleton[:, 0, :2, :] - skeleton[:, 1, :2, :],
                                             axis=1)) * box_scale  # point 0 到point1的平均距离，乘以三可以作为捕捉框的大小
            image_a = []
            crop_boxs = []
            for j in [0, 0, 11, 21]:
                # parameter set
                center_point = (int(frame_width - skeleton[frame_ind, j, 0, 0]), int(skeleton[frame_ind, j, 1, 0]))
                # make crop_box
                x1,y1,x2,y2 = self._create_crop_box(center_point, frame_width, frame_height, int(box_len))
                crop_box=[x1,y1,x2,y2]

                image_a.append(imgs[i][y1:y2, x1:x2])

                # norm crop_box
                norm_crop_box = [crop_box[0]/frame_width,crop_box[1]/frame_height,crop_box[2]/frame_width,crop_box[3]/frame_height]
                crop_boxs.append(norm_crop_box)
            all_crop_boxes.append(crop_boxs)

            local_imgs.append(image_a)

        results['local_imgs']=local_imgs
        results['all_crop_boxes']=all_crop_boxes

        return results


@PIPELINES.register_module()
class ResizeLocalImgs:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional),
    added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "resize_size". Added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear'):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _resize_imgs(self, imgs, new_w, new_h):
        return [
            [mmcv.imresize(
                local_img, (new_w, new_h), interpolation=self.interpolation)
            for local_img in img ]for img in imgs
        ]

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """



        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        try:
            results['local_imgs'] = self._resize_imgs(results['local_imgs'], new_w,
                                                new_h)
        except:
            print('aaa')
        results['local_img_shape'] = (new_h, new_w)
        # results['keep_ratio'] = self.keep_ratio
        # results['scale_factor'] = results['scale_factor'] * self.scale_factor

        assert 'local_imgs' in results

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, ')
        return repr_str


@PIPELINES.register_module()
class CreateCropBoxes:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):

        all_crop_boxes = results['all_crop_boxes']
        crop_quadruple = results['crop_quadruple']
        for i, crop_boxs in enumerate(all_crop_boxes):
            # crop_boxs里的8个clip的左上角坐标按照crop_box左上裁剪的位置移动
            # 如果flip了，crop_box左上的位置会变
            # if not results['flip']:
            # 正常crop
            # 这里是不是不需要左上移动？看看后面用crop_boxs的地方？不，这里需要移动，移动后还需要缩放/
            crop_boxs = [[j[0] - crop_quadruple[0], j[1] - crop_quadruple[1],
              j[2] - crop_quadruple[0], j[3] - crop_quadruple[1]] for j in crop_boxs]
            # else:
            #     flip后crop_quadruple的wh不变，y也不变，就是x变成1-
                # crop_boxs = [[i[0] - (1 - crop_quadruple[0]), i[1] - crop_quadruple[1],
                #               i[2] - (1 - crop_quadruple[0]), i[3] - crop_quadruple[1]] for i in crop_boxs]

            # 缩放
            crop_boxs = [[j[0] / crop_quadruple[2], j[1] / crop_quadruple[3], j[2] / crop_quadruple[2], j[3] / crop_quadruple[3]] for j in
                         crop_boxs]
            all_crop_boxes[i]=crop_boxs
        results['all_crop_boxes'] = all_crop_boxes

        return results



@PIPELINES.register_module()
class FuseLocal:
    """
    如果flip，就把origin_imgs翻转，以及 把skeleton翻转，再提取局部图片
    """

    def __call__(self, results):

        original_imgs = results['original_imgs']
        flip_direction = results['flip_direction']
        flip = results['flip']
        skeleton = results['skeleton']
        if flip:
            # origin imgs flip
            for img in original_imgs:
                mmcv.imflip_(img, flip_direction)
            # skeleton flip
            frame_width = results['original_shape'][1]
            for i in range(results['clip_len']):
                for j in [0, 0, 1, 2, 3, 4, 11, 21]:
                    skeleton[i, j, 0, 0] = int(frame_width - skeleton[i, j, 0, 0])

        results['original_imgs'] = original_imgs
        results['skeleton'] = skeleton

        return results

@PIPELINES.register_module()
class SaveOriginImgs:
    """
    保存imgs后面local处理用
    """

    def __call__(self, results):

        results['original_imgs'] = results['imgs']

        return results

@PIPELINES.register_module()
class NormalizeLocalImgs:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        modality = results['modality']

        if modality == 'RGB':
            n = len(results['local_imgs'])
            local_clip_number = len(results['local_imgs'][0])
            h, w, c = results['local_imgs'][0][0].shape
            local_imgs = np.empty((n, local_clip_number, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['local_imgs']):
                for j, local_clip in enumerate(img):
                    local_imgs[i][j] = local_clip

            for img in local_imgs:
                for local_clip in img:
                    mmcv.imnormalize_(local_clip, self.mean, self.std, self.to_bgr)

            results['local_imgs'] = local_imgs
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str

# formatting


"""Format final imgs shape to the given input_format.
32*8*224*224*3
N_crops x N_clips x L x H x W x C
"""

@PIPELINES.register_module()
class FormatShapeLocalImgs:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False, output_format=''):
        self.input_format = input_format
        self.collapse = collapse
        self.output_format = output_format
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        local_imgs = results['local_imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            local_imgs = local_imgs.reshape((-1, num_clips, clip_len) + local_imgs.shape[1:])
            # -1 x N_clips x L x N_crops x H x W x C
            local_imgs = np.transpose(local_imgs, (0, 1, 3, 6, 2, 4, 5))
            # N_crops x N_clips x C x L x H x W
            local_imgs = local_imgs.reshape((-1,) + (local_imgs.shape[2]*local_imgs.shape[3],) + local_imgs.shape[4:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips

        if self.collapse:
            assert local_imgs.shape[0] == 1
            local_imgs = local_imgs.squeeze(0)

        results['local_imgs'] = local_imgs
        results['local_input_shape'] = local_imgs.shape
        if self.output_format=='imgs':
            results['imgs'] = local_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str

