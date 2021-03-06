---
layout: post
mathjax: true
title: 机器学习第八章-集成学习
categories: [Computer Science]
tags: [ML]
---
### 8.1 个体学习与集成(p171)
* 包含同种个体学习器的集成是同质的，个体学习器称为基学习器。不同学习器的集成是异质的，个体学习器称为组件学习器。
* 为什么集成会比单一学习器有更好的性能呢？需要个体学习器好而不同，具有一定的准确性，并且有一定的多样性。如若假设基分类器错误率相互独立，对于二分类基于Hoeffding不等式可以知道随着分类器数目T的增大，集成错误最终趋于0。

$$
\begin{align}
	P(h_i(x) \ne f(x)) &= \epsilon \\ 
	P(H(x) \ne f(x)) &= \sum_{k=0}^{T/2} \binom Tk \\
	&\le exp(-\frac{1}{2}T(1-2\epsilon)^2)
\end{align}

$$

* 集成学习方法分为两大类，个体学习器存在强依赖关系，必须串型生成的序列化方法，如boosting。以及个体学习器不存在强依赖，可以并行化方法，如bagging和随机森林。

### 8.2 Boosting(p173)
* 工作机制为先从初始训练集训练一个基学习器，再根据其表现对训练样本分布进行调整，使得做错的训练样本在后续新受到更多关注。之后再训练下一个基学习器，直至达到指定的值T，最终将这T个基学习器进行加权结合。
* 典型的AdaBoost算法基于加性模型(additive model)，并最小化指数损失函数

$$
	H(x) = \sum_t^T \alpha_th_t(x)\\
	l_exp(H|D) = E_{x \sim D}[e^{-f(x)H(x)}]
$$

通过求关于H(x)的偏导，可得

$$
	H(x) = \frac{1}{2} ln \frac{P(f(x) = 1 \mid x)}{P(f(x)=-1 \mid x)}
$$

之后，推导出更新公式，此处略。

* 对于数据分布的调整，可以运用重赋权法(re-weighting)实施，即为每个样本重新赋予一个权重。如果学习算法无法赋权，可以用重采样法来处理(re-sampling)。如若因为满足不停止条件，即错误大于0.5，但为达到轮数T，可以继续重采样，重启算法，直到达学习轮数。
* 从偏差-方法分解的角度来看，boosting主要关注降低偏差。

### 8.3 Bagging与随机森林(p178)
#### 8.3.1  Bagging
* 利用自助采样法，分别采样出T个含m个训练样本的采样集，然后基于每个采样集训练出一个基学习器。之后，对所有基学习器，通过简单投票法进行集合。
* 因为自助采样约有36.8%的样本采不到，可以作为验证集做包外估计。对于决策树等，包外估计还可以用作剪枝。
* 从偏差-方差角度看，Bagging主要关心降低方差。

#### 8.3.2 随机森林
* 随机森林每次对决策树的节点随机选择k个属性，再选择一个最优属性用于划分。若k=d。则与传统决策树相同。推荐$$k = log_2d$$。
* 随机森林因为属性扰动，初始性能会较低，但随着学习器数量增大，会达到收敛到更低的泛化误差。同时，因为每轮只是一个属性的子集，训练效率也比一般决策树更高。

### 结合策略(p181)
1. 统计原因:可能有多个假设可以在训练集上达到相同性能，但是如若误选，会导致在测试集上性能不佳。
2. 计算原因:减少陷入局部极小点的风险。
3. 表示的原因:真是假设可能不在当前假设空间，因此结合多个学习器，假设的空间会有所增大。

#### 8.4.1 平均法
* 常用有简单平均法和加权平均，一般在个体学习器差异较大时选择加权平均，否则很容易陷入过拟合。

#### 8.4.2 投票法
* 每个学习器输出一个N维向量，对应每个类别的输出。常用方法如下:
	* 绝对多数投票法(majority voting): 若类别得票超过半数，预测为该类别，否则拒绝预测。
	* 相对多数投票法(plurality voting): 选择得票最高的类别。
	* 加权投票法(weighted voting):每个学习器预测结果多了个权重。
* 硬投票(hard voting)是N维向量将预测类别记为1，其余为0；软投票(soft voting)使用类概率。

#### 8.4.3 学习法
* 当训练数据很多时，常用的结合方法为学习法，即通过另一个学习器来结合初级学习器，称之为次级学习器。Stacking就是典型代表，在k fold 交叉验证上，用其中的一份构建次级学习器学习的训练集。
* 有研究表明，次级学习器用多响应线性回归较好。

### 8.5 多样性(p185)
#### 8.5.1 误差-分歧分解。
* 定义个人学习器与集成学习器的差别为分歧,$$A(h_i \mid x)  = (h_i(x) - H(x))^2$$,平方误差为

$$
	E(h_i \mid x) =  (f(x) - h_i(x))^2\\
	E(H \mid x) = (f(x) - H(x))^2
$$

通过整理和求和，可以得到，
$$
	E=\bar{E} -\bar{A}
$$
其中E为集合学习器误差，$$\bar{E}$$为个体学习器的误差加权和，$$\bar{A}$$为个体学习器的加权分歧值。可见，若个体学习器越精确，分歧值越大，即多样性越大，则集成效果越好。

#### 8.5.2 多样性度量
对于m个样本，二分类任务，分类器$$h_i$$与$$h_j$$的预测结果列联表(contingency table)分别对应预测的样本数目，a为同为正类的数目，d为同为负类的数目，b、c为不一致的数目。 因此可以定义常见的多样性度量:
* 不合度量(disagreement measure)，值域为[0,1]，越大多样性越大

$$
dis_{ij} = \frac{b+c}{m}
$$

* 相关系数(correlation coefficient)，值域为[-1,1]，0为不相关。

$$
	\rho_{ij} = \frac{ad-bc}{\sqrt{(a+b){a+c}(c+d)(b+d)}}
$$

* Q-统计量(Q-statistic)，与相关系数类似。

$$
	Q_{ij} = \frac{ad-bc}{ad+bc}
$$

* k-统计量(k-statistic)，其中p1为分类一致的概率，p2为偶然达成一致的概率。完全一致时为1，偶然一致时为0

$$
	k=\frac{p1-p2}{1-p2}\\
	p1  = \frac{a+d}{m}
	p2 =  \frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}
$$

* k误差图，横轴为k值，纵轴为平均误差。y值越大，个体分类器准确越低，x值越大，学习器多样性越小。

#### 8.5.3 多样性增强
一般思路为引入随机性
* 数据样本扰动，可以从原始数据集中产生不同的数据子集，再训练不同的个体学习器。对不稳定学习器很有效，如决策树，神经网络等，因为训练样本变动易导致学习器有显著变化。不适用于稳定学习器，如线性学习器、支持向量机、朴素贝叶斯、k近邻学习器等。
* 输入属性扰动，随机子空间算法，从初始属性中抽取若干属性的子集，再分别训练学习器。
* 输出表示扰动，有翻转法(Flipping Output)随机改变一些训练样本的标记；输出调制法(Output Smearing)，分类任务转回归
* 算法参数扰动，负相关法，强制个体神经网络使用不同的参数。



