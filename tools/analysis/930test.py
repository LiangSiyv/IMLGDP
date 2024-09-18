import pandas as pd
import os


def read_csv():
    path = r'/home/19031110382/Datas/CSL2018/gloss/colorframes/'
    data = pd.read_csv(r'/home/19031110382/Datas/CSL2018/test_label_slim.csv')
    count = 0
    c_l = []
    for i in data.iloc[:, 0]:
        path_c = (path + i + '/')
        # print(path_c + '\n')
        for file in os.listdir(path_c):
            count = count + 1
        c_l.append(count)
        count = 0
    return c_l


if __name__ == '__main__':
    res = read_csv()
    print(res)
