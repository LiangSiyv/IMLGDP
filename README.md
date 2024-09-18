# IMLGDP Integrated Multi-Local and Global Dynamic Perception Structure for Sign Language Recognition
![fig](https://github.com/LiangSiyv/IMLGDP/blob/main/fig1.png)

## Enviroment Prepare
We use MMAction2 as video understanding tool.
It is a part of the [OpenMMLab](http://openmmlab.org/) project.

The master branch works with **PyTorch 1.5+**.

Below are quick steps for installation.

```
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```
### Create the pretrained folder

Download the pretrained weights given below.

## Train
We utilize 2 Nvidia Tesla V100 GPUs for training.

```python tools/train.py configs/config.py --validate --cfg-options```
## Test
We utilize 1 P100 GPU for testing. Test the trained model with best performance by

```python tools/test.py configs/config.py pretrained/best_result.pth --eval top_k_accuracy```

the trained weights for test on the NMFs\_CSL Dataset can download at  [Google Drive Disk Link](https://drive.google.com/drive/folders/16q1UDJiVZubwJ1fD2tY9-kJs797c0AqG?usp=drive_link) and [Baidu Pan Disk Link](https://pan.baidu.com/).
