# 这是研究生毕设研究的第二种轴承寿命预测算法研究
主要想法：考虑到全生命周期的轴承震动数据过多，希望对数据进行有效压缩（特征提取）之后再进行剩余寿命预测。  
Main Idea: Since the lifecycle data of bearing is too large to train. It is hoped to find out some way to compress the data (extract the feature) and predict the RUL.

# (更新时间线)Update Log

## 20210117
1. 项目提交上传到GitHub
2. 模型的说明、分析及效果请看硕士论文：[滚动轴承剩余寿命预测算法研究及监测软件开发](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1020400323.nh&v=6FTACn1uZ3kFN%25mmd2F1iZry4xN5dw19GEOQ9C%25mmd2BaQuW1cZGqrEFu9Xj6HkonTc%25mmd2BzjzuII)
3. 项目里面有很多可能没用的代码，请不要问我为什么这个模型不行，因为它就是不行。
### English Version
1. Upload the Project to GitHub
2. The description of this porject can be seen in Master Thesis: [滚动轴承剩余寿命预测算法研究及监测软件开发](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CMFD&dbname=CMFDTEMP&filename=1020400323.nh&v=6FTACn1uZ3kFN%25mmd2F1iZry4xN5dw19GEOQ9C%25mmd2BaQuW1cZGqrEFu9Xj6HkonTc%25mmd2BzjzuII). And I am not going to translate it. The latest model is in the file `DIModel2.py`.
3. There may be many useless code (failed models), so do not doubt your result with my code.

## 20200526
1. 用时域未归一化数据可提取出长度为32的特征，但归一化之后就不行
2. 提取出来的特征降维之后大致形成圆形，中间为开始，边缘为结束

## 初建项目
### 需要解决的问题
1. 判断数据是健康还是开始故障
   * 深度学习检测异常值？
   * 特征提取应采用无监督学习——无监督学习的不确定性怎么限制？
2. 如何确保提取的特征具
3. 备想要的信息？以方便rnn进行回归预测
   * rnn可采用注意力机制，将注意力较低的那部分特征认为是没能提取出有用信息？感觉还是不合适
   * rnn根据预测结果反求输入数据的理想值？

