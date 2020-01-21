---
layout: post
mathjax: true
title: 机器学习第三章-线性模型
categories: [Computer Science]
tags: [ML]
---

### 3.1 基本形式(p53)
$$f(x) = w^T+b$$

### 3.2 线性回归(p53)
* 若属性存在序关系，可以用一个连续属性代表离散值，如高、矮 -> {0，1}
* 若属性不存在序关系，需要用k维向量表示，如西瓜、南瓜、黄瓜 -> (0,1,0),(1,0,0),(1,0,0)

1.均方误差(square loss)求解w,b。

$$
\begin{align}
(w^*,b^*) &= argmin_{(w,b)}\sum_{i=1}^{m}{(f(x)-y_i)^2}\\
			&= argmin_{(w,b)}\sum_{i=1}^{m}{(f(x)-wx_i-b)^2}
\end{align}$$

2.用最小二乘法(least square method)求解最小值

$$ \frac{dE_{(w,b)}}{dw} = 2\Bigl( w\sum_{i=1}^{m}{x_i^2} - \sum_{i=1}^{m}{(y_i-b)x_i} \Bigr) $$

$$ \frac{dE_{(w,b)}}{db} = 2\Bigl( mb - \sum_{i=1}^{m}{(y_i-wx_i)} \Bigr) $$

3.对于多元线性回归，如若矩阵列数大于行数，即变量多于样本个数，$$X^TX$$无逆，因此常引入正则化。

### 3.3对数几率回归(p57)
* 对数几率函数(logistic function) $$y = \frac{1}{1+e^{-x}}$$
* 代入后得，$$ln \frac{y}{1-y} = w^Tx+b$$, 其中 $$\frac{y}{1-y}$$ 为几率，反映正例与负例的比例。
* 记$$\beta = (w,b)$$, 对数似然

$$l(w,b) = \sum_i^{m} \ln p(y_i|x_i;w,b)\\
  l(\beta) = \sum_{i}^{m}{(-y_i\beta^Tx_i + \ln(1+e^{\beta^Tx_i}))}
$$

是关于$$\beta$$的高阶可导连续凸函数，可通过凸优化理论，数值优化算法如梯度下降和牛顿法求得其最优解。

### 3.4线性判别分析(Linear Discriminant Analysis)(p60)
* 使得同一类别投影尽可能近，不同类别尽可能远,即最大化

$$J = \frac{||w^T\mu_0 - w^T \mu1||_2^2}{w^T \Sigma_0w + w^T \Sigma_1w} $$

分子为类中心距离，分母为协方差。然后令 $$S_w = \Sigma_0+\Sigma_1, S_b = (\mu_0-\mu_1)(\mu_0-\mu_1)^T$$

$$ J = \frac{w^TS_bw}{w^TS_ww}$$
即LDA优化目标$$S_b S_w$$的广义瑞利商(generalized Rayleigh quotient)，等价于，令$${w^TS_ww} = 1$$

$$\min_{w} w^TS_bw$$

由拉格朗日乘子法，上式等价于

$$ S_bw = \lambda S_w w$$

可得，$$w = S_w^{-1}(\mu_0 - \mu_1)$$。为了更加稳定的数值解，可以再对$$S_w$$做奇异值分解。

### 3.5 多分类学习（p63)
* 可以拆解为1对1(One vs One),1对其余(One vs Rest)和多对多(Many vs Many)
* 纠错输出码(Error Correcting Output Codes, ECOC)
	* 对N个分类器做M次划分，得到M个分类器，M个分类器对测试样例做预测吗，选择与之前编码距离最小的类别作为输出结果。

### 3.6 类别不均衡问题
* 降采样(downsampling),不能简单地扔到样本,可能会损失重要信息,代表算法EasyEnsemble。
* 过采样(oversamping),代表算法SMOTE
