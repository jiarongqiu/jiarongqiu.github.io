---
layout: post
title: 机器学习第五章(p97-120)
categories: [Computer Science]
tags: [ML]
---

### 5.1 神经元模型(p97)
* 超过某一个阈值就兴奋，典型的激活函数是阶跃函数，但由于其不连续、不光滑，实际中替换成sigmoid

### 5.2 感知器与多层神经网络(p98)
* 单层感知器不能解决线性不可分问题，如异或

### 5.3 误差逆传播算法(p101)
* 误差逆传播(BackPropagation, BP)
* 均方误差, $$E_k = \frac{1}{2} \sum_{j=1}^{l}{(\widehat{y}_i^k - y_{j}^k)^2} $$

BP基于梯度下降策略，以目标的负梯度方向对参数进行调整。

$$ \Sigma w_{hj} = -\eta \frac{\partial E_k}{\partial w_{hj}}$$

根据链式法则，

$$  \frac{\partial E_k}{\partial w_{hj}} = \frac{\partial E_k}{\partial \widehat{y}^k_j}  
 \frac{\partial \widehat{y}^k_j}{\partial \beta_j} 
 \frac{\partial \beta_j}{\partial w_{hj}} 
$$

其中 $$w_{hj},\beta_j $$分别为输出层的输入权重和输入值。显然，我们有

$$ \frac{\partial E_k}{\partial \widehat{y}^k_j} = \sum_{j=1}^{l}{(\widehat{y}_i^k - y_{j}^k)}$$

$$ \frac{\partial \widehat{y}^k_j}{\partial \beta_j} =  \widehat{y}^k_j (1- \partial \widehat{y}^k_j)$$

$$ \frac{\partial \beta_j}{\partial w_{hj}} = b_h$$

其中，$$b_h$$为隐藏层的输出，sigmoid的导数为$$f'(x) = f(x)(1-f(x))$$.

* 标准BP为单个样本更新，累积BP是指对整个数据集做更新，往往累积误差下降到一定程度后，下降会比较缓慢，这时候标准BP往往会更快获得更好的解。
* 避免过拟合， 1) 早停; 2) 正则化。

### 5.4 全局最小与局部最小(p106)
为了避免算法陷入局部最小解，常有以下策略避免陷入局部最小解。
1. 按照多种不同参数初始化的网络，按标准训练后，选择其中误差最小的解作为最终参数。
2. 使用模拟退火，在每步中都以一定概率选择比当前更差的结果。接受次优解的概率随时间递减。
3. 使用随机梯度下降，在计算梯度时候加入了随机因素。

上述算法大多为启发式，缺少理论保证。

### 5.5 其他常见神经网络 (p108)

#### 5.5.1 RBF网络
* RBF网络(Radical Basis Function，径向基），被证明足够多隐层神经元可以以任意精度逼近任意连续函数。

$$\phi(x) = \sum_{i=1}^{q}{w_i\rho(x,c_i)}$$

其中,c和w分别对应隐藏单元的中心和权重。

#### 5.5.2 ART网络
* ART(Adaptive Resonance Theory，自适应谐振理论)网络是竞争学习(competitive learning)的重要代表。每一时刻，仅有一个获胜的神经元被激活，其他神经元的状态被抑制。若输入向量和神经元相似度不大于阈值，则重置模块将在识别层增设一个新的神经元。
* ART可以缓解竞争学习的“可塑性-稳定性窘境”，可塑性是指神经网络要有学习新知识的能力，稳定性是指神经网络再学习新知识时要保持对旧只是的记忆。

#### 5.5.3 SOM网络
* 输出层以矩阵方式铺在二维平面内，每个输出单元具有一个权向量。在接收到一个训练样本后，输出单元会计算样本与权向量的距离，距离最近的神经元成为最佳匹配单元。然后又，更新最佳匹配单元及其临近神经元的权向量。

#### 5.5.4 级联相关网络
* 自适应网络，可训练调整网络结构。一般，通过最大化新神经元的输出与网络误差之间的相关性来训练相关的参数。

#### 5.5.5 Elman网络
* 隐藏层输出与下一时刻的输入信号一起作为输入，训练用推广的BP算法，是最常用的递归神经网络之一。

#### 5.5.6 Boltzmann机
* 状态向量s出现的概率讲仅由其能量与所有可能状态向量的能量确定
* 受限Boltzmann机是二分图，先根据输入状态向量计算隐层概率分布，再根据隐层概率分布计算显层。最后，基于此更新连接权重。

### 5.6 深度学习 (p113)
* 深度信念网络(Deep belief network, DBN)
* LeNet
* 深度学习可以看做特征学习(feature learning)或者表示学习(representation learning)。



