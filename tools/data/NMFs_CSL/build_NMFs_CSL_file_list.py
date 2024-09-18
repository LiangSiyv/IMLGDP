# coding: utf-8
import argparse
import glob
import json
import os.path as osp
import random

from mmcv.runner import set_random_seed

from tools.data.anno_txt2json import lines2dictlist
from tools.data.parse_file_list import (parse_directory, parse_diving48_splits,
                                        parse_hmdb51_split,
                                        parse_jester_splits,
                                        parse_kinetics_splits,
                                        parse_mit_splits, parse_mmit_splits,
                                        parse_sthv1_splits, parse_sthv2_splits,
                                        parse_ucf101_splits)
import csv
import fnmatch
import glob
import json
import os
import os.path as osp
import random

# def locate_directory(x):
#     return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))
#
src_folder = '/home/19031110382/Datas/NMFs-CSL_video/jpg_video'
frame_dirs = glob.glob(osp.join(src_folder, '*', '*'))
#
# csv_reader = csv.reader(open('/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video/uni_trainlist01.csv'))
# file_label_map = {file : label for file, label in csv_reader}
# out_path= '/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video'
# train_name = 'NMFs_CSL_train_split_rgb.txt'
#
# for i, frame_dir in enumerate(frame_dirs):
#     lst = os.listdir(frame_dir)
#     total_num = len(lst)
#     if os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]) not in file_label_map.keys():
#         continue
#     with open(osp.join(out_path, train_name), 'w') as f:
#         f.writelines(os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]), total_num,
#           file_label_map[os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1])])
#     print(os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]), total_num,
#           file_label_map[os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1])])




csv_reader = csv.reader(open('/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video/uni_trainlist01.csv'))
file_label_map = {file : label for file, label in csv_reader}

data = []
for i, frame_dir in enumerate(frame_dirs):
    lst = os.listdir(frame_dir)
    total_num = len(lst)
    if os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]) not in file_label_map.keys():
        continue
    data.append([os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]), total_num,
          file_label_map[os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1])]])

random.shuffle(data)
with open('/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video/NMFs_CSL_train_split_rgb.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)



csv_reader = csv.reader(open('/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video/uni_testlist01.csv'))
file_label_map = {file : label for file, label in csv_reader}
out_path= '/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video'
test_name = 'NMFs_CSL_test_split_rgb.txt'

data = []
for i, frame_dir in enumerate(frame_dirs):
    lst = os.listdir(frame_dir)
    total_num = len(lst)
    if os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]) not in file_label_map.keys():
        continue
    data.append([os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1]), total_num,
          file_label_map[os.path.join(frame_dir.split('/')[-2], frame_dir.split('/')[-1])]])

random.shuffle(data)
with open('/home/19031110382/Desktop/19031110382/Datas/NMFs-CSL_video/NMFs_CSL_test_split_rgb.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)