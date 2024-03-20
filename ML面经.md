**1、GBDT算法**
* 无论是处理回归还是分类问题，GBDT使用的决策树都是CART回归树。因为GBDT每次迭代要拟合的是梯度值，是连续值所以要用回归树。损失函数相应的是平方误差。
* 基本思想就是贪心算法，每一轮都最小化损失函数，下一轮在上一轮的残差基础上进行拟合。
  
    ![image](https://github.com/SnoopyXI/-/assets/78628328/923cf47e-2d33-4eed-a4e9-b9fba09f8cda)
  
    ![image](https://github.com/SnoopyXI/-/assets/78628328/51c1a607-b272-48a8-adf8-ff4e13386e88)



**2、LGB和XGB的区别**
