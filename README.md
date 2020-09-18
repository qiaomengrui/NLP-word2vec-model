# word2vec模型
word2vec的整体思想是句子中相近的词之间是有联系的
## 基础（skip-gram 和 CBOW）
### 1、skip-gram的原理
1. skip-gram是利用中心词预测周围window内的词  
2. 内部是通过矩阵计算实现的，如下图
### 2、CBOW的原理 
1. CBOW是利用周围词预测中心词
## 改进（Hierarchical softmax 和 Negative Sampling）
![图片名称](https://github.com/qiaomengrui/NLP-word2vec-model/blob/master/%E5%9B%BE%E7%89%87/skip-gram%E5%8E%9F%E7%90%86%E5%9B%BE.png)  
