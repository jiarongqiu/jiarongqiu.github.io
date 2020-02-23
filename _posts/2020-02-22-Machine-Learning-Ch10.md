---
layout: post
mathjax: true
title: 机器学习第十章-降维与度量学习
categories: [Computer Science]
tags: [ML]
---
$$
\newcommand{abs}[1]{\left| {#1} \right|}
\newcommand{norm}[1]{\left| \left| {#1} \right| \right|_ {2}}
\newcommand{condp}[2]{P({#1} \mid {#2})}
$$

### 10.1 k近邻学习(p225)
* 选择k个最近邻样本投票，得到最终的分类结果。
* 证明其泛化错误率不超过最优分类器的错误率的两倍。 $$P(err) = 1- \sum_{c \in y} \condp{c}{x} \condp{c}{z} $$

$$
\begin{align}
	P(err) &= 1- \sum_{c \in y}\condp{c}{x} \condp{c}{z} \\ 
			&\approx 1-\sum_{c \in y}\condp{c}{x}^2 \\
			&\le 1 - \condp{c^{*}}{x}^2 \\ 
			&= (1+\condp{c^{*}}{x})(1-\condp{c^{*}}{x}) \\
			&\le 1 * (1-\condp{c^{*}}{x})

\end{align}
$$

### 10.2 低维嵌入(p226)
* 高维情形下出现数据样本稀疏、距离计算困难，都被称为维数灾难(curse of dimensionality).解决的重要途径便是降维(dimension reduction).
* 若要求原始空间中样本之间的距离在低维空间中保持一致，即是多维缩放(Multiple Dimensional Scaling). 原始空间距离矩阵$$D \in R^{mxm}$$，降维后的样本空间为$$Z \in R^{d'x m}$$，要满足$$\norm{z_i - z_j} = D_{ij}$$.
为便于讨论，令$$B=Z Z^T$$,降维后的样本Z中心化,即$$\sum_{i}^m z_i = 0$$,则有$$\sum_{i}^m b_{ij} = \sum_j^m b_{ij} = 0$$，则有

$$
\begin{align}
	dist_{ij}^2 &= \norm{z_i}^2 + \norm{z_j}^2 - 2 z_i^T z_j \\ 
				&= b_{ii} + b_{jj} -  2b_{ij} \\ 
\end{align}
$$

$$
\begin{align}
	\sum_{i}^m dist_{ij}^2 &= tr(B) + mb_{jj} \\
	\sum_{j}^m dist_{ij}^2 &= tr(B) + mb_{ii} \\
	\sum_{i}^m \sum_{j}^m dist_{ij}^2 &= 2mb_{jj} \\
	

\end{align}
$$

则 $$b_{ij} = -\frac{1}{2}(dist_{ij}^2-dist_{i.}^2-dist_{.j}^2+dist_{..}^2)$$,即得到使降维前后距离不变的內积矩阵B

对B做奇异值分解（eigenvalue decomposition), B = $$VAV^T$$,Z=$$A^{1/2}V^T$$,可除去为0的特征值或较小的特征值，使得距离尽可能相等，但不完全相等。

### 10.3 主成分分析(p229)
* 主成分分析(Prinicipal Componenet Analysis),考虑一个超平面对所有样本有恰当的表达。具有以下性质：
	* 最近重构性: 样本点到这个超平面的距离都足够近
	* 最大可分性: 样本点在这个超平面上的投影尽可能分开
* 二者具有等价推导，先考虑最近重构性。 样本点$$x_i$$在超平面上的坐标是$$z_{ij} = w_j^Tx_i$$。若基于z去重构在源空间的样本点$$\hat{x_i} = \sum_i^{d'}z_{ij}w_j$$。这样考虑整个训练集的距离，有

$$
	\sum_i^m \norm{\sum_i^{d'} z_{ij}w_j - x_i}^2 = \sum_i^m z_i^T z_i - 2 \sum_i^m z_i^T W^Tx_i + const \\
					\approx -tr(W^T(\sum_i^m x_ix_i^T)W)
$$

进而有了主成分分析的优化目标

$$
	\min_W -tr(W^TXX^TW),\\
	s.t W^TW =1
$$

* 从最大可分性出发，使所有样本点的投影方差最大化，因为有数据中心化，所以方差为$$\sum_i W^T x_i x_i^T W$$,优化目标可以写成

$$
	\max_W tr(W^TXX^TW) \\ 
	s.t. W^TW = I
$$

利用拉格朗日乘子法并求导，有$$XX^Tw_i = \lambda w_i$$.于是，可以对$$X^TX$$进行特征值分解，并基于特征值排序，选择前d'个特征值。降维后的维度d'要提前指定，可以选择下式中成立的最小d'值

$$
	\frac{\sum^{d'} \lambda_i}{\sum^{d} \lambda_i } \ge 0.95
$$

一方面，降维舍弃了一部分信息，但使得样本的采样密度增大，另一方面，当数据受到噪声影响时，较小的特征值往往噪声。

### 10.4 核化线性降维(p232)
* 核主成分分析(Kernelized PCA),考虑非线性降维的情况，即假定$$z_i = \phi(x_i)$$,则有

$$
	(\sum_i^m z_i z_i^T) w_j = \lambda_j w_j \\
	(\sum_i^m \phi(x_i) \phi(x_i)^T) w_j = \lambda_j w_j
$$

则有, $$w_j = \sum_i^m \phi(x_i) \alpha_i^j$$，其中$$\alpha_i^j = \frac{1}{\lambda_j} z_i^T w_j$$. 由于不清楚$$\phi$$的形式，引入核函数 $$K(x_i,x_j) = \phi(x_i)^T \phi(x_j)$$,化简后可得

$$
	K\alpha^j = \lambda_j \alpha^j
$$

之后，便化为了特征分解的问题，取K最大的d'个特征值对应的特征向量即可。对于新样本，其投影后的坐标为

$$
	z_j = w_j^T \phi(x) = \sum_i^m \alpha_i^j \phi(x_i)^T \phi(x_i) \\
			= \sum_i^m \alpha_i^j K(x_i,x_j)
$$



### 10.5 流形学习(p234)

#### 10.5.1 等度量映射
等度量映射(Isometric Mapping)，直接在高维空间计算直线距离是具有误导性的，低维嵌入流形上两点间的距离是测地距离。因为局部具有欧式空间同胚的性质，所以对每个点，基于欧式空间找出其近邻点，然后就能建立一个近邻连接图。
于是，计算两点之间的测地线距离，就转变为了近邻连接图上最短路径问题，可以通过Dijkstra算法或者floyd算法求解。在得到了新的距离矩阵之后，可以通过MDS算法求解在低维空间的投影。

#### 10.5.2 局部线性嵌入
局部线性嵌入(Locally Linear Embedding)希望能保留邻域内样本间的线性关系,假定样本点$$x_i$$的坐标能通过他的领域样本$$x_j,x_k,x_l$$的坐标通过线性组合而重构出来，即

$$
	x_i = w_{ij}x_j + w_{ik}x_k + w_{il}x_l
$$

LLE先为每个样本$$x_i$$找到其近邻下标集合$$Q_i$$，然后计算出基于$$Q_i$$中的样本点对$$x_i$$的重构系数

$$
	\min_{w_1,w_2,...,w_m} \sum_i^m \norm{x_i - \sum_{j \in Q_i} w_{ij}x_j}^2 \\
		s.t. \sum_{j \in Q_i} w_{ij} = 1
$$

有闭式解

$$
	w_{ij} = \frac{\sum_{k \in Q_i} C_{jk}^{-1}}{\sum_{l,s \in Q_i} C_{ls}^{-1}}\\
	C_{jk} = (x_i-x_j)^T(x_i-x_k)
$$

低维空间的坐标可通过下式求解

$$
	\min_{z_1,z_2,...,z_m} = \sum_i^m \norm{z_i-\sum_{j \in Q_i} w_{ij} z_j}^2 \\
$$

等价于, 

$$
\min_Z tr(ZMZ^T) \\ 
		s.t. ZZ^T = 1 \\
	M = (I-W)^T(I-W)
$$

M最小的d'个特征值对应的特征向量组成的矩阵即为Z

### 10.6 度量学习(p237)
度量学习(metric learning),就是直接尝试学习出一个合适的距离度量。下面先定义一个灵活，科学系的距离度量，假定每个属性重要性不同，以及属性之间是有关的。由此，可以定义一个半正定对称矩阵，就得到了
马氏距离(Mahalanobis distance)

$$
	dist_{mah}{x_i,x_j} = (x_i-x_j)^T M (x_i,x_j)
$$

下面便需要为M设置一个目标，假定我们希望提高分类器性能，则可以将M直接嵌入到近邻分类器的评价指标中去，通过优化该性能指标相应地求得M。

* 可以用近邻分类器的正确率作为优化目标求得M，也可以对必连，勿连的优化问题，让必连的距离较小，勿连的距离较大，进而也可以求得M。若M是一个低秩矩阵，则通过对M进行特征值分解，可以得到一个降维矩阵$$P \in R^{d \times rank(M)}$$














