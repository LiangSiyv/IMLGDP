# IMLGDP Integrated Multi-Local and Global Dynamic Perception Structure for Sign Language Recognition
![fig](https://github.com/LiangSiyv/IMLGDP/blob/main/fig1.png)
## Train
We utilize 2 Nvidia Tesla V100 GPUs for training.

```python tools/train.py configs/config.py --validate --cfg-options```
## Test
We utilize 1 P100 GPU for testing. Test the trained model with best performance by

```python tools/test.py configs/config.py best_result.pth --eval top_k_accuracy```

the trained weight on the NMFs\_CSL Dataset can download at  [Google Drive Disk Link]() and [Baidu Pan Disk Link]().
