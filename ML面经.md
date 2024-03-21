**1、LGB和XGB的区别与联系**
* 两者都是GBDT的具体解决方案。GBDT的基本思想是通过下一轮拟合上一轮的残差，但如果损失函数非凸，第一步我们就不容易计算一个最优解，从而计算损失。所以一般最开始会采用初始化值（均值或比例）进行迭代，而后将损失函数的负梯度作为残差的近似值进行拟合。
* LGB：
  * LightBoost以叶子结点为单位进行构造，每次LightGBM只选择分裂增益最大的叶子节点进行分裂，直到最后达到最大叶子数 ![image](https://github.com/SnoopyXI/-/assets/78628328/c2d0d29f-f9b0-44b9-b5f8-4b68312207c4)
  * LGBM同时对连续特征使用了直方图算法，基本思想是：先把连续的浮点特征值离散化成k个整数，同时构造一个宽度为k的直方图。在遍历数据的时候，根据离散化后的值作为索引在直方图中累积统计量，当遍历一次数据后，直方图累积了需要的统计量，然后根据直方图的离散值，遍历寻找最优的分割点。这样做有两个优点：一是内存占用更小，计算代价更小；二是利用差加速可以用非常小的代价获得兄弟叶子的直方图。
* XGB：
  * 在XGBoost中，决策树每次以层为单位进行构造，直到达到最大深度 ![image](https://github.com/SnoopyXI/-/assets/78628328/64c19f41-82ae-45cf-baff-0f550e9a75b6)

**2、常见的各类函数计算公式**
* 信息论
  * 信息熵： $H(Y)=-\displaystyle\sum^n_{i=1}p(y_i)\log_2p(y_i)$
  * 条件熵： $H(Y|X) = \displaystyle\sum_{x \in X}p(x)H(Y|X=x)$
  * 信息增益/互信息： $IG(Y, X) = H(Y) - H(Y|X)$
  * 基尼系数： $Gini(p) = \displaystyle\sum^K_{k=1}p_k(1-p_k)$
  * KL散度/相对熵： $D_{KL}(p||q) = \displaystyle\sum^N_{i=1}p(x_i)(\log p(x_i) - \log q(x_i)) = E[\log p(x_i) - \log q(x_i)$
