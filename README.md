# HMM中文分词模型
## 概述
使用numpy实现HMM模型进行中文分词。

`hmm_segment.py`中定义的`HMMSegment`类包含“初始状态概率向量”、“状态转移概率矩阵”和“观测概率矩阵（发射矩阵）”的计算，HMM模型的测试（计算准确率、召回率、F1值）和HMM模型分词方法。

`main.py`作为程序的入口。

## 其他
数据集：icwb2-data，完整数据集[下载](https://github.com/yuikns/icwb2-data)

HMM的基本原理参考：[NLP硬核入门-隐马尔科夫模型HMM](https://zhuanlan.zhihu.com/p/87632700)

Viterbi算法原理参考：[图解Viterbi维特比算法](https://zhuanlan.zhihu.com/p/63087935)