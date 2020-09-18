# word2vec模型
word2vec的整体思想是句子中相近的词之间是有联系的
## 基础（skip-gram 和 CBOW）
### 1、skip-gram的原理
1. skip-gram是利用中心词预测周围window内的词  
2. 内部是通过矩阵计算实现的，如下图  
![skip-gram](pic\skip-gram.png)
### 2、CBOW的原理
1. CBOW是利用周围词预测中心词
## 改进（Hierarchical softmax 和 Negative Sampling）
### 1、Hierarchical softmax：通过其将维度V变为log<font size=2>2</font>V，采用哈夫曼树的思想将softmax转换成计算log<font size=2>2</font>V个sigmoid
