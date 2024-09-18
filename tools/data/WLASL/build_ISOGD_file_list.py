# coding: utf-8
import csv
import random

file = '/home/19031110382/Desktop/19031110382/Datas/ISOGD/test.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)


random.shuffle(data)
with open('/home/19031110382/Desktop/19031110382/Datas/ISOGD/ISOGD_test_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)

file = '/home/19031110382/Desktop/19031110382/Datas/ISOGD/train.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)

random.shuffle(data)
with open('/home/19031110382/Desktop/19031110382/Datas/ISOGD/ISOGD_train_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)

file = '/home/19031110382/Desktop/19031110382/Datas/ISOGD/valid.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append(line)


random.shuffle(data)
with open('/home/19031110382/Desktop/19031110382/Datas/ISOGD/ISOGD_valid_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)

