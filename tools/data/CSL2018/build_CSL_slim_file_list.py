# coding: utf-8
import csv
import random
file = '/home/19031110382/Datas/CSL2018/train_label_slim.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append([line[0]+'.mp4',line[1]])
random.shuffle(data)
with open('/home/19031110382/Datas/CSL2018/CSL_slim_train_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)

file = '/home/19031110382/Datas/CSL2018/test_label_slim.csv'
data = []
with open(file, 'r') as f:
    result = csv.reader(f)
    for line in result:
        data.append([line[0]+'.mp4',line[1]])
random.shuffle(data)
with open('/home/19031110382/Datas/CSL2018/CSL_slim_test_list_videos.txt', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(data)

