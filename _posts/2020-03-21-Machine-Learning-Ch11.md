---
layout: post
mathjax: true
title: 机器学习第十一章-特征选择与稀疏学习
categories: [Computer Science]
tags: [ML]
---
$$
\newcommand{abs}[1]{\left| {#1} \right|}
\newcommand{norm}[2]{\left| \left| {#1} \right| \right|_ {#2}}
\newcommand{condp}[2]{P({#1} \mid {#2})}
$$

### 11.1 子集搜索与评价(p247)
* 特征选择(feature selection)的原因
	* 维数灾难
	* 去除不相关特征往往会降低学习任务的难度
* 冗余特征有时候可作为中间概念降低学习难度，因此不可去除
* 特征选择的过程
	* 子集搜索(subset search)，贪心的添加或者删减特征，有前向搜索，后向搜索和双向搜索。
	* 子集评价(subset evaluation),A为属性子集，D为数据集，$$D^v$$为数据集根据A划分成的数据子集，每个里面的属性在A上取值相同。

$$
Gain(A) = Entropy(D) - \sum_{v=1}^V \frac{\abs{D^v}}{\abs{D}} Entropy(D^v)
$$

### 11.2 过滤式选择(p249)
Relief(Relevant Feature)设计相关统计量来度量特征的重要性，选择相关统计量大于阈值$$\tau$$的特征作为重要特征。对每个样本$$x_i$$，寻找其最近猜对近邻(near-hit)和猜错近邻(near-miss)。其中，diff对于离散型属性，若相等则为；反之，为1。若对于连续性属性，计算其L1距离。

$$
\delta^j = \sum_i -diff(x_i^j,x_{i,nh}^j)^2 + diff(x_i^j,x_{i,nm}^j)^2
$$

可以看到，对于某一个属性，若$$x_i$$与猜对近邻的距离小于猜错近邻的距离，则证明这个属性是有益的； 反之，则是有负作用的。最后，根据所有样本的平均，可以计算出各属性的统计分量。

Relief是针对二分类任务设计的，其扩展变体Relief-F能处理多分类任务。若$$x_i$$属于第k类，则在第k类样本中寻找一个最近邻$$x_{i,nh}$$，作为猜中近邻。然后在其余类别各寻找一个最近邻，记做猜错近邻$$x_{i,l,nm}, (l=1,2...., l \ne k)$$。$$p_l$$为类别l的占比。

$$
\delta^j = \sum_i -diff(x_i^j,x_{i,nh}^j)^2 + \sum_{l \ne k}(p_l * diff(x_i^j,x_{i,l,nm}^j))^2
$$

### 11.3 包裹式选择(p250)
* 相比过滤式选择，会考虑之后的学习器的性能，效果比过滤式选择更好，但是计算开销会更大。
* LVW(Las Vegas Wrapper),基于Las Vegas Method，不同于蒙特卡洛法
	1. 设置时间T作为循环次数上限
	2. 随机生成特征子集A‘，
	3. 若其交叉验证的误差E'好于之前，更新t=0,E=E',$$A^* = A$$;反之，回到2，并自增t=t+1
	4. 输出$$A^* $$


### 11.4 嵌入式选择与L1正则化(p252)
* LASSO, Least Absolute Shrinkage and Selection Operator，L1范数会比L2更易于获得稀疏解。可以分别考虑L1和L2的等值线。L1的等值线是矩形，与坐标轴相交位置为极值点；L2的等值线是原型，各处loss相等，无极值点。从梯度角度，L2正则化对应的梯度见效速率更快，且趋近于0。

$$\min_w \sum_i^m (y_i - w^T x_i)^2 + \lambda \norm{w}{1}$$

* 可以通过PGD(Proximal Gradient Descent)求解

### 11.5 稀疏表示与字典学习(p254)
* 將稠密的特征变成稀疏表示(sparse representation)。
* 字典学习（dict learning),学习目标如下。其中B为字典矩阵，第一项是为了更好的重构x，第二项是为了让表示尽可能稀疏。对于下式，也可以交替求解，先固定B求解$$\alpha_i$$,再求解B。

$$
\min_{B,\alpha_i} \sum_{i=1}^m \norm{x_i-B\alpha_i}{2}^2 + \lambda \sum_{i=1}^m \norm{\alpha_i}{1}
$$

### 11.6 压缩感知(p257)
* 压缩感知(compressed sensing)
* 矩阵补全(matrix completion)

$$
	\min_X rank(X), s.t. (X)_{ij} = (A)_{ij},ij是已观测信号
$$






