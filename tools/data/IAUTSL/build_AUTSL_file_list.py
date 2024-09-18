# coding: utf-8
import csv
import random

file = '/home/19031110382/liangsiyu1/Datas/TAUSL/uni_test_labels.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/liangsiyu1/Datas/TAUSL/AUTSL_test_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)


file = '/home/19031110382/liangsiyu1/Datas/TAUSL/uni_train_labels.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/liangsiyu1/Datas/TAUSL/AUTSL_train_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)


file = '/home/19031110382/liangsiyu1/Datas/TAUSL/uni_val_gt.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/liangsiyu1/Datas/TAUSL/AUTSL_valid_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)
