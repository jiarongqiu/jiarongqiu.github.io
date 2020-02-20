---
layout: post
mathjax: true
title: 机器学习第九章-聚类
categories: [Computer Science]
tags: [ML]
---
$$
\newcommand{abs}[1]{\left| {#1} \right|}
\newcommand{condprob}[2]{P({#1} \mid {#2})}
$$
### 9.1 聚类任务(p197)
根据样本的特征将数据划分为若干个不相交的子集，称为簇(Cluster)。常用聚类算法寻找数据内在的分布结构。

### 9.2 性能度量(p197)
* 物以类聚，希望同一簇的样本彼此接近，不同簇的样本尽可能不同，即簇内相似度(intra-cluster similarity)高，簇间相似度(inter-cluster similarity)低。
* 如果度量有标签，即有参考模型，称之为外部指标;如果直接观察聚类结果而不利用任何参考模型，称为内部指标。
* 对于有参考模型，考虑样本对$$(x_i,x_j)$$，a为二者在聚类算法和参考模型中均同类，b为在聚类算法中同类，但在参考模型中异类，c为聚类异类，参考模型种同类，d为在聚类和参考模型中均异类。a+b+c+d = m(m-1)/2，由此可以导出以下的外部指标。
	* Jaccard系数

		$$ JC = \frac{a}{a+b+c}$$

	* FM指数

		$$ FMI = \sqrt{\frac{a}{a+b} * \frac{a}{a+c}}$$

	* Rand指数

		$$ RI = \frac{a+d}{m(m-1)}$$

	显然上述指标结果均在[0,1]区间，且越大越好。
* 对于内部指标，我们需要观察avg(C)类内平均聚类，diam(C)为类内最远距离,$$d_{min}(C_i,C_j)$$为类间最短聚类,$$d_cen(C_i,C_j)$$为两类的类中心距离。
	* DB指数

		$$ DBI = \frac{1}{k}\sum_{i=1}^{k}\max_{j \neq i}(\frac{avg(C_i)+avg(C_j)}{d_{cen}(\mu_i,\mu_j)})$$

	* Dunn指数

		$$ DI  = \min_{1\le i \le k}\min_{j\neq i}(\frac{d_{min}(C_i,C_j)}{\max_{1\le l \le k} diam(C_l)})$$

	显然，DBI越小越好，DI越大越好。

### 9.3 距离计算(p199)
* 若是$$dist(x_i,x_j)$$一个距离度量，需要满足一些基本性质
	* 非负性,$$dist(x_i,x_j)\ge0$$
	* 同一性,$$dist(x_i,x_j)=0$$当且仅当$$x_i = x_j$$
	* 对称性,$$dist(x_i,x_j)= dist(x_j,x_i)$$
	* 直递性,$$dist(x_i,x_j)\le  dist(x_i,x_k) + dist(x_k,x_j)$$

* 闵可夫斯基距离(Minkowski Distance)。

	$$dist_{mk}(x_i,x_j) = (\sum_{u=1}^{n}\abs{x_{iu}-x_{ju}}^p )^\frac{1}{p}$$

	p=2时，为欧式距离；p=1时，为曼哈顿距离。

* 在讨论距离计算时，需关注属性是否定义了序关系。例如{1,2,3}上计算距离，1与2比1与3接近。所以，显然闵可夫斯基距离可用于有序属性。对于无序属性，如{飞机，汽车，火车}，则可采用VDM(Value Difference Metric)。

* 令$$m_{u,a}$$表示在属性u上取值为a的样本数,$$m_{u,a,i}$$表示为第i个样本簇中在属性u上取值为a的样本数，k为样本簇数，则在属性u上两个离散值a，b的VDM距离为

	$$VDM_p(a,b) = \sum_{i=1}^k\abs{\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}}^p$$

* 因此对于混合属性，可以结合闵可夫斯基距离和VDM距离。假定有$$n_c$$个有序属性,$$n-n_c$$个无序属性，则

	$$MinkovDM_p(x_i,x_j)= (\sum_{u=1}^{n_c}\abs{x_{iu}-x_{ju}}^p + \sum_{u=n_c+1}^n  VDM_p(x_{iu},x_{ju}) )^\frac{1}{p}$$

* 有时度量相似性的距离未必一定要满足距离度量的所有性质，尤其是直递性。比如，希望人，马分别和人马相近，但是人和马要不相似，这就与直递性矛盾。这样的距离称为非度量距离(non-metric distance)。也有时，有必要基于数据样本来确定合适的距离计算式，这可以通过距离度量学习实现。

### 9.4 原型聚类(p202)
基于原型的聚类(prototype-based clustering),算法先对原型进行初始化，然后迭代地更新求解。

#### 9.4.1 k均值算法
* k均值(k-means)算法针对聚类所得簇划分C最小化平方误差

$$E =  \sum_{i=1}^k  \sum_{x \in C_i}\left|  \left| x-\mu_i \right|  \right|^{2}_{2}   $$

1. 先随机选择k个样本作为初始化均值向量
2. 对每个样本计算到每类中心的距离，并划归到最近的簇内
3. 重新计算均值向量
4. 重复上述过程，直到均值向量未更新

#### 9.4.2 学习向量量化
学习向量量化(Learning Vector Quantization)假设数据样本带有类别标记。每个原型向量代表一个类(可以有重叠)
1. 每个原型向量通过其类别中的一个随机样本初始化
2. 从样本集中随机选取样本$$(x_i,y_i)$$,计算与各原型向量的距离
3. 找到最近的原型向量，若类别相同,则更新原型向量$$p' = p + \eta (x_j - p_{i^*})$$; 反之，更新$$p' = p - \eta (x_j - p_{i^*})$$
4. 执行上述过程直到满足条件

直观上看，样本跟原型向量同类，则使其更接近；异类，则使其更远。
* 在求得一组原型向量后，可以对样本根据其最近的原型向量做划分，通常称为Voronoi 剖分(tessellation)

#### 9.4.3 高斯混合聚类
* 概率密度函数, $$p(x) = \frac{1}{(2\pi)^{\frac{n}{2}}\abs{\Sigma}^{\frac{1}{2}}} e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$
* 定义高斯混合分布, $$P_M(x) = \sum_{i=1}^k \alpha_i \condprob{x}{\mu_i,\Sigma_i}, \sum_{i=1}^k \alpha_i = 1$$
* 假定样本的生成过程由高斯混合模型给出：首先，根据$$\alpha$$定义的先验分布的概率值选择高斯混合成分；然后，根据被选择的混合成分的概率密度分布函数采样，从而生成对应的样本。
* 若训练集由上述过程生成，令随即变量$$z_j \in {1,2,...,k}$$表示生成样本$$x_j$$的高斯混合成分。显然，可由贝叶斯定理求出其后验概率$$\gamma_{ji}$$，即样本$$x_j$$由第i个高斯混合成分生成的后验概率。


$$
	\begin{align}
	\condprob{z_j=i}{x_j} &= \frac{P(z_j = i) \condprob{x_j}{z_j=i}}{p_M(x_j)}\\
						&= \frac{\alpha_i \condprob{x_j}{\mu_i,\Sigma_i}}{\sum_{l=1}^k \alpha_l \condprob{x_j}{\mu_l,\Sigma_l}}
	\end{align}
$$

* 当高斯混合分布已知时，选择后验概率最大的簇作为样本$$x_j$$的类别标记。
* 接着，便需要估计模型参数了，显然，对于样本集D，可以用极大似然估计，即最大化对数似然

$$
\begin{align}
	LL(D) &= ln(\prod_{j=1}^m p_M(x_j))\\
		&= \sum_{j=1}^m ln(\sum_{i=1}^k \alpha_i \condprob{x_j}{\mu_i,\Sigma_i})
\end{align}
$$

* 关于$$\mu$$求导，有

$$
\sum_{j=1}^m \frac{\alpha_i \condprob{x_j}{\mu_i,\Sigma_i}}{\sum_{l=1}^k\alpha_l \condprob{x_j}{\mu_l,\Sigma_l}}(x_j-\mu_i) = 0\\
\sum_{j=1}^m \gamma_{ji}(x_j-\mu_i) = 0 \\
\mu_i = \frac{\sum_{j=1}^m \gamma_{ji} x_j}{\sum_{j=1}^m \gamma_{ji}}\\
$$

* 关于$$\Sigma$$求导，有

$$
\Sigma_{i} = \frac{\sum_{j=1}^m \gamma_{ji}(x_j-\mu_i)(x_j-\mu_i)^T}{\sum_{j=1}^m\gamma_{ji}}
$$

* 另外还要注意，$$\sum_i\alpha_i = 0$$，因此需要考虑LL(D)的拉格朗日形式,经过求导求和可知

$$\alpha_i = \frac{1}{m} \sum_{j=1}^{m} \gamma_{ji}$$

* 由上述推导即可获得高斯混合模型的EM算法。E步，根据当前参数计算后验概率$$\gamma_{ji}$$；M步再根据上述等式，通过$$\gamma_{ji}$$更新模型参数$$(\alpha_i,\mu_i,\Sigma_i)$$

### 9.5 密度聚类(p211)
从样本密度的角度来考察样本之间的可连接性，并基于可连接样本不断扩展聚类簇，以获得最终的聚类结果。
* 定义核心对象:$$x_i$$的 $$\epsilon$$邻域的样本数大于MinPts;密度直达:$$x_j$$ 位于$$x_i$$的$$\epsilon$$邻域，且$$x_i$$是核心对象，则称为$$x_j$$由$$x_i$$密度直达;密度可达:考虑密度直达的传递性，即$$x_j$$由$$x_j$$由一系列密度直达的样本相连。

1. 先寻找核心对象的集合
2. 随机选择一个核心对象，找到所有样本中其密度可达的样本，构成簇
3. 循环上述过程，直到所有核心对象被访问

### 9.6层次聚类(p214)
层次聚类(hierarchical clustering)是树形的聚类结构，有自底向上或自顶向下的分拆策略。
* AGNES是自底向上的策略。
	1. 将数据集中每个样本看做一个初始簇
	2. 每步找出距离最近的两个簇合并，距离
		1. 最小距离,$$d_{min}(C_i,C_j) = \min_{x \in C_i,y \in C_j}dist(x,y)$$，被称为单链接(single linkage)
		2. 最大距离,$$d_{min}(C_i,C_j) = \max_{x \in C_i,y \in C_j}dist(x,y)$$，被称为全链接(complete linkage)
		3. 平均距离,$$d_{avg}(C_i,C_j) = \frac{1}{\abs{C_i}\abs{C_j}}\sum_{x \in C_i}\sum_{y \in C_j}dist(x,y)$$，被称为均链接(average linkage)
	3. 重复上述过程，直到达到指定簇的个数













