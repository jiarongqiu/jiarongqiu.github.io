---
layout: post
mathjax: true
title: 机器学习第六章(p121-145)
categories: [Computer Science]
tags: [ML]
---
$$
\newcommand{sumim}{\sum_{i=1}^{m}}
\newcommand{sumjm}{\sum_{j=1}^{m}}
$$

### 6.1间隔与支持向量(p121)
空间中划分超平面$$w^Tx + b = 0$$，任意一个样本点到平面的距离为$$ r = \frac{ \| w^Tx+b\| }{\|w\|}$$.假设分类正确，我们有

$$\begin{cases}
w^T x_i+ b \ge +1,y=+1 \\
w^T x_i+ b \le -1,y=-1 \\
\end{cases}$$

使等号成立的样本点叫做支持向量，两个异类的支持向量到超平面的距离之和为

$$ \gamma = \frac{2}{\|w\|}$$

欲寻找最大间隔，即

$$ \max_{w,b} \frac{2}{\|w\|} \\
s.t.  y_i(w^Tx_i + b) \ge 1, i= 1,2, ... ,m $$

等价于 $$\min_{w,b} \frac{1}{2} \|w\|^2$$。这便是SVM(Support Vector Machine)的基本型了。

### 6.2对偶问题(p123)
* 分界面模型 $$f(x) = w^Tx+b$$
* 拉普拉斯乘子法的直观解释，约束的梯度和函数的梯度必出于同一方向，否则总可以沿着梯度增加的方向移动，得到更大的函数值。因此对于约束g，有$$\nabla f = \lambda \nabla g$$。
* 拉格朗日函数可写成如下，

$$L(w,b,\alpha) = \frac{1}{2} \| w\|^2 + \sumim\alpha_i(1-y_i(w^Tx_i+b))$$

* 根据拉格朗日乘子法，分别对w和b求偏导，有以下等式

$$ 
w = \sumim \alpha_i y_i x_i \\
0 = \sumim \alpha_i y_i \\
$$

* 接着可以将原式转为为其对偶问题，

$$
	\max_\alpha \sumim\alpha_i - \frac{1}{2} \sumim \sum_{j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j\\ 
	s.t. \sumim \alpha_iy_i = 0, \\
		\alpha_i \ge 0 ,i =0,1,2,....m
$$

* 因为有不等式约束，上述过程还需要有KKT条件(Karush-Kuhn-Tucker)，即求，

$$
\begin{cases}
\alpha_i \ge 0 \\
y_i f(x_i) - 1 \ge 0 \\
\alpha_i(y_i f(x_i) - 1) = 0 \\ 
\end{cases}
$$

于是，对于任意一个样本，总有$$\alpha_i = 0 $$ 或 $$ y_i f(x_i) = 1$$。于是，若$$\alpha_i = 0$$ 则样本不会对f(x)产生影响;若$$ y_i f(x_i) = 1$$，则他们均为支持向量。

* SMO算法: 每次先固定$$\alpha_i \alpha_j$$以外的参数，然后求解对偶问题得到更新后的参数值。另外，选取的参数违背KKT条件的程度越大，更新后目标函数的提升值就越大。因此，先选取一个违背程度大的，另一个再启发式地选取一个样本之间间隔最大的参数。

* 对于偏移量b，现实中往往可根据$$y_sf(x_s) = 1$$,选取所有支持向量求解的平均值。

$$ b = \frac{1}{\left|S\right|} \sum_{s \in S}(1/y_s - \sum_{i \in S}\alpha_i y_i x_i^T x_s)$$

### 6.3 核函数(p126)

$$
\newcommand{\projectx}{\phi(x)}
\newcommand{\projectxt}{\phi(x)^T}
\newcommand{\projectxit}{\phi(x_i)^T}
\newcommand{\projectxi}{\phi(x_i)}
\newcommand{\projectxj}{\phi(x_j)}
$$

* 当训练样本线性不可分时，可以把它映射到更高维的特征攻坚，去寻找一个划分超平面。令$$\projectx$$表示将x映射后的特征向量，于是$$f(X) = w^T\projectx + b$$。以同样的方式，可以得到其对偶问题。

$$\max_\alpha \sumim \alpha_i - \frac{1}{2} \sumim \sumjm \alpha_i \alpha_j y_i y_j \projectxit \projectxj\\
	s.t. \sumim \alpha_i y_i = 0,\\
	\alpha_i \ge 0,i=1,2,3....m
$$

* 定义核函数 $$K(x_i,x_j) = <\projectxi,\projectxj> = \projectxit \projectxj$$，即$$x_i,x_j$$在特征空间上的内机等于它们在原式样本空间中通过核函数计算的结果，这样便不用直接计算高维空间的内积结果了。

* 当特征空间及对应的映射不可知时，我们只能定义核函数，隐式地定义这个特征空间。同时，我们有定理主要一个对称函数所对应的核矩阵半正定，那么就可以当做核矩阵使用。
* 常用的核函数有线性核($$x_i^T x_j$$),高斯核($$exp(-\frac{\| x_i-x_j\|^2}{2\sigma^2})$$)等。

### 6.4 软间隔与正则化(p129)

* 因为不确定特征空间是否线性可分，以及不确定在特征空间中是否存在过拟合，为此，要引入软间隔的概念(soft margin)，即允许部分样本被分类错误。当然，在最大化间隔的同时，分类错误，不满足约束的样本要尽可能少。于是，优化目标可写为

$$\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m l_{0/1}(y_i(w^Tx_i+b)-1)\\
其中，l_{0/1}是0/1损失函数 \begin{cases}1,if z<0\\0,otherwise\end{cases}
$$

* 因为0/1损失函数非凸，且不连续，常用代替损失(surrogate loss) 去拟合，如
	* hinge损失(max(0,1-z))
	* 指数损失(exp(-z))
	* 对率损失(log(1+exp(-z)))
* 若采用hinge损失，原式可变为

$$\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m max(0,(y_i(w^Tx_i+b)-1))$$

引入松弛变量$$\xi_i\ge 0$$，又可变为

$$\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i \\
s.t. y_i(w^Tx_i + b) \ge 1-\xi_i\\
\xi_i \ge 0, i=1,2,3....m
$$

* 通过拉格朗日乘子法，同样的可以得到其对偶问题

$$
\max_\alpha \sumim \alpha_i - \frac{1}{2} \sumim \sumjm \alpha_i \alpha_j y_i y_j x_i^T x_j\\
s.t. \sumim \alpha_i y_i = 0\\
0 \le \alpha_i \le C, i =1,2,3....m
$$

* 根据KKT条件，此时若$$\alpha_i\gt 0$$, 则必有$$y_if(x_i) = 1 - \xi_i$$。若$$\alpha_i\lt C$$ 则样本落在支持向量上；若$$\alpha_i = 0$$,则$$\xi_i$$可以落在最大间隔内部，也可以被错误分类，进而实现了软间隔支持。

* 优化目标的共性，一般第一项用来描述划分平面间隔的大小，另一项描述与训练集的误差。

$$
\min_{f} \Omega(f) + C \sumim l(f(x_i),y_i)
$$

其中， $$\Omega(f)$$被称为结构风险(structural risk)，用于描述模型的特征；第二项成为经验风险(empirical risk)，用于描述模型与训练数据的契合度。
从经验风险最小化的角度来看，$$\Omega(f)$$对应了希望获得复杂度较小的模型，另一方面，该信息可以削减假设空间，从而降低最小化训练误差的过拟合风险，对应正则化问题。
$$\Omega(f)$$被称作正则化项，C为正则化常数。其中L2番薯倾向于w的取值均匀，即非零分量个数稠密，L0和L1倾向于尽可能稀疏，非零分量个数尽量少。

### 6.5 支持向量回归(p133)
$$
\newcommand{\relaxi}{\xi_i}
\newcommand{relaxihat}{\hat{\xi_i}}
$$
* 对于f(x)与y之间差值，只有当大于$$\epsilon$$时，我们才计算其损失值。于是，SVR问题可形式化为

$$\min_{w,b} \frac{1}{2} \|w\|^2 + C \sumim l_c(f(x_i)-y_i)$$

其中C为正则化常数,$$l_c$$ 为不敏感损失, 
$$
l_c(z) = 
\begin{cases}
0, if \left| z \right| \le \epsilon\\
\left| z \right| - \epsilon, otherwise\\
\end{cases}
$$

* 引入松弛变量$$\relaxi,\relaxihat$$，原式可改写为

$$ \min_{w,b,\relaxi,\relaxihat} \frac{1}{2} \|w\|^2 + C\sumim(\relaxi + \relaxihat)\\
s.t. f(x_i) -y_i \le \epsilon + \relaxi \\ 
y_i - f(x_i) \le \epsilon + \relaxihat\\
\relaxi \ge 0, \relaxihat \ge 0 ,i = 1,2,3...m$$

* 仍然可以通过之前类似的拉格朗日乘子法求得其对偶问题。基于KKT条件，可以看出当且仅当样本在$$\epsilon$$带之外时，$$\alpha$$可以为非零。SVR的解可以写成

$$
\begin{align}
f(x) &= \sumim (\hat{\alpha_i} - \alpha_i) x_i^T x +b\\
	&= \sumim  (\hat{\alpha_i} - \alpha_i) k(x,x_i) + b
\end{align}
$$

### 6.6 核方法(p137)
* 表示定理(representer theorem), 令H为核函数k对应的再生核希尔伯特空间，$$\|h\|_H $$ 
表示H空间中关于h的范数，对于任意单调递增函数$$\Omega:[0,\infty] \to R$$和任意非负损失函数，优化问题

$$
	\min_{h \in H} F(h) = \Omega(\|h\|_H) + l(h(x_1),h(x_2),.....,h(x_m))
$$

的解总可以写成
$$h(x) = \sumim \alpha_i k(x,x_i)$$

* 将线性判分析进行非线性拓展，得到核线性判别分析。推导此处略。








