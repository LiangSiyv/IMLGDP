# Auther: Liang Siyu
# @Time     : 2022/9/30 10:13
# @Auther   : Liang Siyu
# @Email    : 358682594@qq.com
import os
import torch
from torchvision import transforms


def tensor_to_jpg_and_save(filename, img):
    tensor_size = list(img.shape)
    # 建立目录文件
    if not os.path.exists(filename):
        os.mkdir(filename)
    # 根据输入tensor建立文件夹

    ToPIL = transforms.ToPILImage()
    for i in range(tensor_size[0]):
        if not os.path.exists(filename + '/{}'.format(i)):
            os.mkdir(filename + '/{}'.format(i))
        for j in range(tensor_size[1]):
            # if not os.path.exists(filename + '/{}'.format(i) + '/{}'.format(j)):
            #     os.mkdir(filename + '/{}'.format(i) + '/{}'.format(j))
            for k in range(tensor_size[2]):
                # img tensor转化成.jpg并保存
                # img = torch.randn(3, tensor_size[3], tensor_size[4])  # channel数为[0]索引下的3
                pics = ToPIL(img[i][j][k])
                pics.save(filename + '/{}'.format(i) + '/{}_{}.jpg'.format(j, k))


if __name__ == '__main__':
    img = torch.randn([2, 24, 32, 224, 224])
    filename='filename'
    tensor_to_jpg_and_save(filename, img)
