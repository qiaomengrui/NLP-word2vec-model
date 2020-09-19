import numpy as np
from collections import defaultdict

# 超参数
settings = {
    'window_size': 2,  # 窗口尺寸 m
    'n': 300,  # 单词嵌入(word embedding)的维度,维度也是隐藏层的大小。
    'epochs': 50,  # 表示遍历整个样本的次数。在每个epoch中，我们循环通过一遍训练集的样本。
    'learning_rate': 0.01  # 学习率
}

class word2vec():

    def __init__(self):
        """参数设置"""
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, settings, corpus):
        """得到训练数据"""
        # defaultdict(int)  一个字典，当所访问的键不存在时，用int类型实例化一个默认值
        word_counts = defaultdict(int)

        # 遍历语料库corpus
        for row in corpus:
            for word in row:
                # 统计每个单词出现的次数
                word_counts[word] += 1

        # 词汇表的长度(去重)
        self.v_count = len(word_counts.keys())
        # 词汇表->list
        self.words_list = list(word_counts.keys())
        # 以词汇表中单词为key，索引为value的字典数据
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # 以索引为key，以词汇表中单词为value的字典数据
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        # 记录目标词和对应周围词的 one - hot
        training_data = []

        for sentence in corpus:
            # 每一段词个数
            sent_len = len(sentence)
            # 给每个词配个i
            for i, word in enumerate(sentence):
                # 给出当前中心词 one-hot 编码
                w_target = self.word2onehot(sentence[i])
                # 用于记录当前 w_target 所有的周围词 one-hot
                w_context = []
                # 取中心词前方和后方
                for j in range(i - self.window + 1, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        # 将周围词 one-hot 加入 w_context
                        w_context.append(self.word2onehot(sentence[j]))
                # 将目标词和对应周围词的 one-hot 加入 training_data
                training_data.append([w_target, w_context])
        return np.array(training_data)

    def train(self, training_data):
        """训练"""
        # 随机化参数w1,w2
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        # 迭代epochs次数
        for i in range(self.epochs):
            # 损失loss
            self.loss = 0

            # w_t（目标词one-hot向量） -> w_target 词向量, w_c ->w_context 向量
            for w_t, w_c in training_data:
                # 前向传播
                y_pred, h, u = self.forward(w_t)
                # 计算误差
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                # 反向传播，更新参数
                self.backprop(EI, h, w_t)
                # 计算总损失
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

    def word2onehot(self, word):
        """将word转为onehot"""
        word_vec = [0 for i in range(0, self.v_count)]
        word_vec[self.word_index[word]] = 1
        return word_vec

    def forward(self, x):
        """前向传播"""
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    def softmax(self, x):
        """softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def backprop(self, e, h, x):
        """ 反向传播 """
        d1_dw2 = np.outer(h, e)
        d1_dw1 = np.outer(x, np.dot(self.w2, e.T))
        self.w1 = self.w1 - (self.lr * d1_dw1)
        self.w2 = self.w2 - (self.lr * d1_dw2)

    def word_vec(self, word):
        """获取词向量"""
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def vec_sim(self, word, top_n):
        """找相似的词"""
        # 测试词的向量
        v_w1 = self.word_vec(word)
        # 存放所有
        word_sim = {}
        # 对表每个词进行向量计算
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            # 利用测试词向量与词汇表词向量计算
            theta_sum = np.dot(v_w1, v_w2)
            # 计算每个单词与测试词的相似概率 （np.linalg.norm(v_w1) 求范数 默认为2范数，即平方和的二次开方）
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den
            word_sim[self.index_word[i]] = theta
        # 根据相似度进行排序
        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)
        # 找相似度大的前top_n词
        for word, sim in words_sorted[:top_n]:
            print(word, sim)

    def get_w(self):
        """获取所有词向量"""
        w1 = self.w1
        return w1

# 数据准备
text = "natural language processing and machine learning is fun and exciting"
# 按照单词间的空格对我们的语料库进行分词
corpus = [[word.lower() for word in text.split()]]
# 初始化一个word2vec对象
w2v = word2vec()
# 数据准备
training_data = w2v.generate_training_data(settings, corpus)
# 训练
w2v.train(training_data)

"""测试"""
# 获取词的向量
word = "machine"
vec = w2v.word_vec(word)
print(word, vec)
# 找相似的词
w2v.vec_sim("machine", 3)
