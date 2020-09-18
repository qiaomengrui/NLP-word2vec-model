# word2vec模型
word2vec的整体思想是句子中相近的词之间是有联系的
## 一、基础（skip-gram 和 CBOW）
### 1、skip-gram的原理
1. skip-gram是利用中心词预测周围window内的词  
2. 内部是通过矩阵计算实现的，如下图  
![skip-gram](pic\skip-gram.png)
### 2、CBOW的原理
1. CBOW是利用周围词预测中心词
## 二、改进（Hierarchical softmax 和 Negative Sampling）
### 1、Hierarchical softmax（层次softmax）
1. 通过其将维度V变为log2V，采用哈夫曼树的思想将softmax转换成计算log<font size=2>2</font>V个sigmoid
### 2、Negative Sampling（负采样）
1. 得到一个正样本概率和几个负样本概率，共K+1个样本，效果比多分类好
## 三、优化（subsampling of frequent words）
### 1、subsampling of frequent words（重采样）
1. 通过将词频大的，被删除的概率大，词频小的，被删除的概率小，原因是出现次数多的词往往含有的信息少，例如："a"、"the"
## 四、模型复杂度
模型复杂度是通过模型训练时所使用的参数个数来评估的，通过计算得到各模型复杂度
NNLM：Q=N\*D+N\*D\*H+H\*log2V  
RNNLM：Q=H\*H+H\*log2V  
CBOW+HS：Q=N\*D+D\*log2V  
skip-gram+HS：Q=C(D+D\*log2V)  
CBOW+NEG：Q=N\*D+D\*(K+1)  
skip-gram+NEG：Q=C(D+D\*(K+1))  
## 五、超参数
利用genism做word2vec时，其中dim一般选择100-500之间，min_count一般选择2-10之间，如果数据集少则将min——count调为1


