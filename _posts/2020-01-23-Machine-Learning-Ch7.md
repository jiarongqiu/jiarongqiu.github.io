---
layout: post
mathjax: true
title: 机器学习第七章-贝叶斯分类器
categories: [Computer Science]
tags: [ML]
---
$$\newcommand{sumjn}{\sum_{j=1}^N}
\newcommand{argmax}[1]{\underset{#1}{\operatorname{argmax}}}
\newcommand{condprob}[2]{P({#1} \mid {#2})}
\newcommand{abs}[1]{\left| {#1} \right|}
$$
### 7.1 贝叶斯决策论(p147)
* 贝叶斯决策论(Bayesian decision theory)，样本x上的条件风险为$$R(c_i \mid x) = \sumjn \lambda_{ij} P(c_j\mid x)$$,即把样本分为$$c_j$$类的损失。其中

$$\lambda_{ij} = \begin{cases} 0,ifi=j\\ 1,otherwise \end{cases}$$

* 不难看出，要想获得$$P(c \mid x)$$，可以有两种方式。一是直接建模，由此获得的是判别模型(discriminative models);另一种是生成模型(generative models)。显然，决策树，BP和SVM都是前者。而对于生成式模型，我们有

$$
\begin{align}
P(c \mid x) &= \frac{P(x,c)}{P(x)} \\ 
			&= \frac{P(x \mid c) P(c)}{P(x)} 
\end{align}
$$

其中P(c)是先验，表达了样本空间中各类样本所占比例;P(x)是用于归一化的证据银子；$$P(x \ mid c)$$是类条件概率，或者称为似然(likelihood),因为很多样本取值在训练集没有出现，所以直接用频率去估计是显然不可行的。

### 7.2 极大似然估计(p149)
* 估计类条件概率的常用策略是先假定其具有某种确定的概率分布，再基于训练样本对其进行估计，实际上此过程是参数估计的过程。对于此，统计学界有两种学派，频率主义学派认为参数虽然未知，但却是固定值；贝叶斯学派则认为参数是未观察到的随即变量，其本身也有分布。
* 此处介绍频率主义学派的极大似然估计(Maximum Likihood Estimation，简称MLE)。

$$ P(D_c \mid \theta_c) = \prod_{x \in D_c} P(x \mid \theta_c) $$

为了防止连乘下溢出，一般我们采用对数似然，转换上式为连加。

* 对于正态分布$$p(x \ mid c) ~ N(\mu_c, \sigma_c^2)$$，则参数的极大似然估计为

$$ 
\hat{\mu_c} = \frac{1}{\left| D_c \right|} \sum_{x \in D_c}{x},\\
\hat{\mu_c} = \frac{1}{\left| D_c \right|} \sum_{x \in D_c}{(x-\hat{\mu_c})(x - \hat{\mu_c})^T}
$$

### 7.3 朴素贝叶斯分类器(p150)
* 假设属性条件独分布

$$
P(c \mid x) = \frac{P(c)P(x \mid c)}{P(x)} = \frac{P(c)}{P(x)} \prod_{i=1}^{d}{P(x_i \ mid c)}
$$

* 因此，判定准则为

$$
h_{nb}(x) =  \argmax{c \in Y}P(c) \prod_{i=1}^d \condprob{x_i}{c}
$$

* 训练过程,$$P(c) = \frac{\abs{D_c}}{\abs{D}}，\condprob{x_i}{c} = \frac{\abs{D_{c,x_i}}}{\abs{D_c}}$$。对于连续样本，则用正态分布去估计，其中正态的均值和方差分别取第c类样本在第i个属性上的均值和方差。

* 因为判定准则为连乘，如果因为某一个属性和类别的关联在训练集没出现过，就否定所有属性的关联，显然不合理。因此，在估计概率值需要进行平滑，常用拉普拉斯修正(Laplacian Correction)。具体来说对训练过程修正为

$$
\begin{align}
P(c)’ &= \frac{ \abs{D_c} + 1}{\abs{D_c} + N} \\ 
\condprob{x_i}{c}' &=  \frac{\abs{D_{c,x_i}}+1}{\abs{D_c}+N}\\
\end{align} 
$$

### 7.4 半朴素贝叶斯分类器(p154)

* 修改假设为独依赖估计，每个属性最多仅依赖其他一种属性，问题就转化为确定每个属性的父属性。最直接的方法是假设所有属性都依赖同一个属性，超父(super-parent)，然后通过交叉验证等模型选择方法来确定超父，由此形成了SPODE方法。
* TAN方法则是在最大带权生成图上的基础上通过以下步骤确定属性依赖关系。
	1. 计算任意两个属性之间的条件互信息(conditional mutual information),$$I(x_i,x_j \mid y) = \sum_{x_i,x_j;c\in Y} P(x_i,x_j \mid c) log \frac{P(x_i,x_j \mid c)}{P(x_i \mid c) P(x_j \mid c)}$$
	2. 以属性为节点构建完全图，图上的权重即为$$I(x_i,x_j \mid y)$$
	3. 构建最大带权生成树
	4. 加入类别节点y,增加y到各个属性的有向边。

* AODE(Averaged One-Dependent Estimator)是基于集成学习的分类器，尝试将每个属性作为超父来构建SPODE，然后再选择有足够数据支撑的作为最终结果。

### 7.5 贝叶斯网络(p156)
* 贝叶斯网(Bayesian network)又称信念网("belief network")，他借助有向无环图来刻画属性之间的依赖关系，并使用条件概率表来描述属性的联合概率分布。具体来说，贝叶斯网B由结构G和参数$$\theta$$组成。

#### 7.5.1 结构
* 贝叶斯网络假设每个属性与它的非后裔属性独立，同时有三种典型的依赖关系，同父，V型和顺序结构。在同父结构中，若父节点的取值已知，则子节点条件独立。在顺序结构中，若中间节点值已知，则头与尾独立。对于V型结构，若子节点已知，则父节点们比不独立；反之，独立。
* 为了分析条件独立性，可以将有向图转为无向图，即找出所有V型结构，连接V型结构的父节点，同时将有向图全部转换为无向图。由此产生的无向图称为道德图(moral graph)，相连的过程被称为道德化(moralization)。

#### 7.5.2 学习
* 贝叶斯网的学习过程是基于评分搜索，定义一个评分函数来评估贝叶斯网与训练数据的契合程度。常用的评分函数基于信息论准则，应选择综合编码长度最短的贝叶斯网，这就是“最小描述长度”准则。
* 给定训练集D，贝叶斯网B在D的评分函数为

$$
s(B \mid D) = f(\theta)\abs{B} - LL(B \mid D)
$$

其中$$\abs{B}$$是贝叶斯网的参数个数，$$f(\theta)$$表示描述每个参数$$\theta$$所需要的字节数，而

$$
LL(B \mid D) = \sum_{i=1}^m log P_B(x_i)
$$

是贝叶斯网的对数似然。显然，第一部分计算的是编码B所需要的字节数，第二项计算概率分布对D描述的好坏。

#### 7.5.3 推断
* 基于贝叶斯网精确推断是NP难的，因为往往使用吉布斯采样来进行近似推断。
* 令$$Q= \{Q_1,Q_2...Q_n\}$$表示待查询的变量，$$E=\{E_1,E_2,....,E_k\}$$为证据变量，目标是计算$$\condprob{Q=q}{E=e}$$。比如，Q为{好瓜，甜度}，E为{色泽，敲声，根蒂}，且已知其值。具体吉布斯采样方法如下:
	1. 先随机产生一个与证据E=e一致的样本
	2. 每次从当前样本出发，对非证据变量进行采样，改变其值，采样概率根据贝叶斯网络B和其他变量的当前取值计算获得
	3. 计算后验概率为$$\frac{n_q}{T}$$
实际上，吉布斯采样是在E=e的子空间中进行随机漫步。每一步仅依赖前一步的状态，是一个马尔科夫链。在一定条件下，无论从什么初始状态,马尔科夫链收敛于一个平稳分布。
* 若贝叶斯网中存在极端概率0或者1，则不能保证马尔科夫链存在且分布平稳。

### 7.6 EM算法(p162)
* 在实际应用中往往会遇到不满证的训练样本，有些属性变量未被观察到。未观测的变量的学名是隐变量"latent variable"。令X表示已观测变量，Z表示隐变量集，$$\theta$$表示模型参数。若用极大似然似然估计，则

$$
LL(\theta \mid X,Z) = ln P(X,Z\mid \theta)
$$
* E步(Expectation)，以当前参数推断隐变量分布$$\condprob{Z}{X,\theta^t}$$，并计算关于Z的期望

$$
Q(\theta \mid \theta^t) = E_{Z \mid X,\theta^t} LL(\theta \mid X,Z) 
$$

* M(Maximization)步，寻找参数最大化期望似然,即

$$
\theta^{t+1} = \argmax{\theta} Q{\theta \mid \theta^t}
$$

* 简单来说，EM算法使用了两个交替计算，第一步是期望E步，利用当前估计的参数值来计算对数似然的期望值；第二步是最大化(M)步，寻找能使E步产生的似然期望最大化的参数值。交替执行前两步直至收敛。