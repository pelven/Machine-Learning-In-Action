# -*- coding: utf-8 -*-
"""
鸢尾花数据预处理并绘制散点图

@author: pelven
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


csvFile = open(".//DataSet//iris.csv")
lines = csv.reader(csvFile)
next(lines)

data = []
species = []
labels = []

#鸢尾花共有setosa,versiclor,virginica三类，将其类别转换为数字标签
le = preprocessing.LabelEncoder()
le.fit(["setosa", "versicolor", "virginica"])

#读取数据集
for line in lines:
    lineArr = []
    #四种特征，sepal_length,sepal_width,petal_length,petal_width
    for i in range(4):
        lineArr.append(np.float32(line[i]))
    data.append(lineArr)
    species.append(line[4])
    #将类别转换为数字标签
    labels = le.transform(species)  
    
#按特征归一化，即数据按列归一化
min_val = np.min(data, 0)
max_val = np.max(data, 0)
ranges = max_val - min_val
data_rows, data_cols = np.shape(data)
min_val = np.tile(min_val, (data_rows,1))
ranges = np.tile(ranges, (data_rows,1))
data_norm = (data - min_val) / ranges

#数据可视化,绘制散点图
type0_x = []
type0_y = []
type1_x = []
type1_y = []
type2_x = []
type2_y = []
axes = plt.subplot(111)
m = 0  #sepal_length
n = 1  #sepal_width
for i in range(len(labels)):
    if labels[i] == 0:  # setosa
        type0_x.append(data_norm[i][m])
        type0_y.append(data_norm[i][n])

    if labels[i] == 1:  # versicolor
        type1_x.append(data_norm[i][m])
        type1_y.append(data_norm[i][n])

    if labels[i] == 2:  # virginica
        type2_x.append(data_norm[i][m])
        type2_y.append(data_norm[i][n])

type0 = axes.scatter(type0_x, type0_y, s=20, c='red')
type1 = axes.scatter(type1_x, type1_y, s=20, c='green')
type2 = axes.scatter(type2_x, type2_y, s=20, c='blue')
plt.xlabel("sepal_length/cm")
plt.ylabel("sepal_width/cm")
axes.legend((type0, type1, type2), ("setosa", "versiclolor", "virginica"), loc=2)



















