# coding: utf-8
import csv
import random

file = '/home/19031110382/Datas/WLASL2000/uni_test_2000.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/Datas/WLASL2000/WLASL2000_test_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)
file = '/home/19031110382/Datas/WLASL2000/uni_val_2000.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/Datas/WLASL2000/WLASL2000_valid_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)
file = '/home/19031110382/Datas/WLASL2000/uni_train_2000.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/Datas/WLASL2000/WLASL2000_train_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)
