# -*- coding: utf-8 -*-
"""
Linear Regression 线性回归
以鸢尾花前三个特征为输入，第四个特征为输出
@author: pelven
"""

import csv
import numpy as np
#import matplotlib.pyplot as plt

##读取鸢尾花数据集,并划分为训练集和验证集
def loadData():
    train_x = []
    train_y = []
    validation_x = []
    validation_y = []
    
    csvFile = open(".//DataSet//iris.csv")
    lines = csv.reader(csvFile)
    next(lines)
    
    x = []
    y = []
    #读取数据集
    for line in lines:
        lineArr = []
        #四种特征，sepal_length,sepal_width,petal_length,petal_width
        for i in range(3):
            lineArr.append(np.float32(line[i]))
        x.append(lineArr)
        y.append([np.float32(line[3])])
        
    #将数据随机打乱以获得更好的训练效果
    x = np.array(x)
    y = np.array(y)
    state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(state)
    np.random.shuffle(y)
    
    #将70%的数据划分为训练集，将30%的数据作为验证集
    #data_rows, data_cols = np.shape(data)  
    train_x = x[0:135,:]
    train_y = y[0:135,:]
    validation_x = x[135:,:]
    validation_y = y[135:,:]
    
    return train_x, train_y, validation_x, validation_y

#利用线性回归的方法训练模型
def train(train_x, train_y, validation_x, validation_y):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    validation_x = np.mat(validation_x)
    validation_y = np.mat(validation_y)
    
    training_steps = 100000 #训练次数
    learning_rate = 0.00001  #学习率
    #初始化权重与偏置
    weights = np.ones((3,1), dtype=np.float32)
    weights = np.mat(weights)
    biases = np.float32(1)
    for i in range(training_steps):
        y_ = train_x * weights + biases
        train_error = train_y - y_
        weights[0,0] = weights[0,0] + learning_rate * (np.sum(np.multiply(train_error,train_x[:,0])))
        weights[1,0] = weights[1,0] + learning_rate * (np.sum(np.multiply(train_error,train_x[:,1])))
        weights[2,0] = weights[2,0] + learning_rate * (np.sum(np.multiply(train_error,train_x[:,2])))
        biases = biases + learning_rate * (np.sum(train_error))
        #当前参数下验证集损失
        if ((i+1)%1000 == 0) or (i==0):
            y = validation_x * weights + biases
            validation_error = validation_y - y
            loss = np.sum(np.multiply(validation_error,validation_error))
            print("train steps: %d, validation loss: %g"%(i+1,loss))
            
    return weights, biases

##主程序入口
def main(argv=None):
    train_x, train_y, validation_x, validation_y = loadData()
    weights, biases = train(train_x, train_y, validation_x, validation_y)
    return weights, biases
    
if __name__ == '__main__':
    main()