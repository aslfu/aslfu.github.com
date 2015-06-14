---
layout: post
title: "感知机模型及简单算法实现"
date: 2015-06-14 20:52:35 +0800
comments: true
categories: 
---

### 感知机模型

假设输入空间（特征空间是）$\mathcal{X}\subseteq\mathcal{R}^{n}$,输出空间是$\mathcal{Y}=\{+1,-1\}$.由输入空间到输出空间的如下函数称为感知机。  

<!--more-->  
${f}(x)=sign(w\bullet x+b)$

----

### 感知机策略  

假设训练数据集是线性可分的，感知机学习的目标是求得一个能够将训练集正负样本完全分开的分离超平面。为了找出这样的超平面，即确定感知机模型参数$w,b$，需要确定一个**学习策略**，即定义损失函数，并将其极小化.

损失函数的一个自然选择是误分类点的总数，但这样的损失函数不是参数$w,b$的连续可导函数，不易优化。损失函数的另一个选择是误分类点导超平面的总距离。定义为： 
 
$L(w,b)＝-\sum_{x_{i}\in{M}}{y_{i}(w\bullet x_{i}+b)}$  

---

### 感知机算法

感知机学习算法是误分类驱动的，具体采用随机梯度下降法。首先，任意选取一个超平面$w_{0},b_{0}$,然后用梯度下降法不断地极小化目标函数。极小化过程中不是一次使M中所有误分类点的梯度下降，而是一次随机选取一个误分类点。

$ \nabla_{w}L(w,b)=-\sum_{x_{i}\in{M}}y_{i}x_{i}$

$ \nabla_{w}L(w,b)=-\sum_{x_{i}\in{M}}y_{i}$

随机选取一个误分类点$(x_{i},y_{i})$,对$w,b$进行更新：

$w<-w+\eta y_{i}x_{i}$

$b<-b+\eta y_{i}$

---

### Python实现

~~~ python
# coding:utf-8
import numpy as np
import pandas as pd
"""
简单感知机学习算法

感知机模型:f(x)=sign(w*x+b)
算法策略:损失函数:L(w,b)=y(w*x+b)
学习算法:梯度下降法
w = w + eta*y*x
b = b + eta*y

数据来源：
《统计学习方法》 李航 例2.1 P29
@Python
"""


class Perceptron:
    """Perceptron

    Methods
    ---------

    __init__: 构造函数
    isError: 判断是否为误分类点
    updateWeights: 更新参数
    train: 训练数据


    """
    def __init__(self, eta, w0, b0, iterMax, data):
    	"""
        eta: 学习率，w0：权重向量w的初始值，b0:偏置b的初始值，iterMax:最大循环次数
        """
        self.eta = eta
        self.w = w0
        self.b = b0
        self.iterMax = iterMax
        self.data = data

    def isError(self, record):
        if (np.dot(self.w, record[0:-1]) + self.b)*record[-1] > 0:
            return False
        else:
            return True

    def updateWeights(self, record_err):
        self.w = self.w + self.eta * record_err[0:-1] * record_err[-1]
        self.b = self.b + self.eta * record_err[-1]
        return

    def train(self):
        n = len(self.data)
        flag = True  # Ture 仍包含误分类点；False 没有误分类点
        iterNum = 0
        """
        停止条件：
        训练集中没有误分类点 或 循环次数超限
        """
        while flag and iterNum < self.iterMax:
            iterNum += 1
            for i in range(n):
                if self.isError(data.values[i]):
                    self.updateWeights(data.values[i])
                    flag = True
                else:
                    flag = False
        return (self.w, self.b)

if __name__ == "__main__":
    data = pd.DataFrame()
    data["x1"] = np.array([3, 4, 1])
    data["x2"] = np.array([3, 3, 1])
    data["y"] = np.array([1, 1, -1])
    n = len(data)
    p = Perceptron(1, np.zeros(2), 0, 10, data)
    result = p.train()
    print str(result)
~~~