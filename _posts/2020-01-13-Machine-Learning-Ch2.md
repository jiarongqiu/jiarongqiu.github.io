---
layout: post
mathjax: true
title: 机器学习第二章-模型评估与选择
categories: [Computer Science]
tags: [ML]
---

### 2.1 经验误差与过拟合(p23)
* training error, empiracal error 在训练集上的误差
* generalization error 在新样本上的误差

### 2.2 评估方法(p24)
#### 2.2.1 留出法 
* hold out,一般$$\frac{2}{3}$$-$$\frac{4}{5}$$做训练
* 根据类别分层采样

#### 2.2.2 交叉验证
* k-fold cross validation: 用分层采样，分成k等份
* 留一法(eave-one-out):评估准确，但计算代价过大

#### 2.2.3 自助法
* 自助法(bootstapping):对m个样本的数据集D，对其采样m次，放到D‘中。D’中可能会出现多次重样样本。拿D‘作为训练集，D/D’作为测试集，约为1/3。适用于数据集较小，不易划分的情况。
 
#### 2.2.4 调参与最终模型
* 基于验证集去选择模型和调参

### 2.3 性能度量(p28)
#### 2.3.1 错误率与精度
* 精度(accuracy):分类正确样本数/样本总数

#### 2.3.2 查准率、查全率和F1
* 查准率(precisioin):$$\frac{TP}{TP+FP}$$
* 查全率(recall):$$\frac{TP}{TP+TN}$$
* PR曲线:根据预测结果进行排序，逐个样本计算recall与precision，绘制曲线
* 平衡点(BEP:Break-Even Point):当recall=precision时的值
* F1:$$F1 = \frac{2 * P * R}{P+R}$$。一般形式$$F_\beta = \frac{(1+\beta^2) * P * R}{(\beta^2 * P)+R}$$

#### 2.3.3 ROC与AUC
* ROC 纵轴真正例率 $$TPR = \frac{TP}{TP+FN}$$ 横轴假正例率$$\frac{FP}{FP+TN}$$。也是先排序，再计算。最理想情况y=1
* AUC 为ROC曲线下面积

#### 2.3.4 代价敏感错误率与代价曲线
* 代价敏感(cost-sensitve)错误率:$$E = \frac{a * cost}{m}$$
* 代价曲线:根据ROC里面的(TPR,FPR)画线段(0,FPR)到(1,FNR)

### 2.4 比较检验(p37)

#### 2.4.1 假设检验 
* 检验单个学习器，用t检验

#### 2.4.2 交叉t检验
* 两个学习器对比

#### 2.4.3 McNemar检验
#### 2.4.4 Frediman 与 Nemenyi后续检验
1. 用 Frediman在单一数据集上对多个算法进行差别显著性的检验
2. 用Nemenyi去判断两两算法间是否有显著性差别

### 2.5 偏差与方差(p44)

* 对算法泛化性能的拆解,f(x;D)为训练集输出，$$y_D$$为训练集标签
* 方差(variance):$$var(x) = E_D\left[(f(x;D) - \bar{f}(x))^2\right]$$，使用样本数不同的训练集产生的方差
* 噪声:$$\epsilon^2 = E_D\left[(y_D-y)^2\right]$$，数据标签和真实标签的差异
* 偏差(bias): $$ bias^2(x) = (\bar{f}(x)-y)^2 $$，期望输出与真实的差异
* 分解 $$ E(f;D) = bias^2(x) + var(x) + \epsilon^2$$

其中偏差度量了学习算法的期望预测与真实结果的差异，代表着算法的拟合能力; 方差度量了不同数据集数据扰动造成的影响;噪音定义了问题本身的难度。

在训练过程中，先是偏差主导，之后方差主导。当二者平衡时，泛化误差最小。



