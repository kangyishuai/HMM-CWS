import pickle
from collections import Counter

import numpy as np


class HMMSegment(object):
    """隐马尔可夫分词模型。"""

    _words = []
    _states = []
    _vocab = set([])
    _puns = set(r"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                r"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■")

    def __init__(self, config, mode=""):
        self.config = config

        if mode == "train":
            train_corpus_path = config.get("train_corpus_path")
            self._read_corpus_from_file(train_corpus_path)
            self._gen_vocabs()
        elif mode == "test":
            pass
        else:
            self.states, self.init_p = self._get_init_state()
            self.trans_p = self._get_trans_state()
            self.vocabs, self.emit_p = self._get_emit_state()

    def _is_puns(self, c):
        """判断是否是符号。"""
        return c in self._puns

    @staticmethod
    def _read_state_from_file(state_path):
        """读取文件。"""
        return pickle.load(open(state_path, "rb"))

    @staticmethod
    def _save_state_to_file(content, path):
        """保存文件。"""
        pickle.dump(content, open(path, "wb"))

    def _read_corpus_from_file(self, file_path):
        """读取语料。"""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            self._words.extend(
                [word for word in line.strip().split(" ") if word and not self._is_puns(word)]
            )

    def _gen_vocabs(self):
        """生成词典。"""
        self._vocab = list(set(self._words)) + ["<UNK>"]

    def _word_to_states(self, word):
        """词对应状态转换。"""
        word_len = len(word)
        if word_len == 1:
            self._states.append("S")
        else:
            state = ["M"] * word_len
            state[0] = "B"
            state[-1] = "E"
            self._states.append("".join(state))

    def _cal_init_state(self):
        """计算初始概率。"""
        init_counts = {"S": 0.0, "B": 0.0, "M": 0.0, "E": 0.0}
        for state in self._states:
            init_counts[state[0]] += 1.0
        words_count = len(self._words)
        init_state = {k: (v+1)/words_count for k, v in init_counts.items()}

        return init_state

    def _cal_trans_state(self):
        """计算状态转移概率。"""
        trans_counts = {
            "S": {"S": 0.0, "B": 0.0, "M": 0.0, "E": 0.0},
            "B": {"S": 0.0, "B": 0.0, "M": 0.0, "E": 0.0},
            "M": {"S": 0.0, "B": 0.0, "M": 0.0, "E": 0.0},
            "E": {"S": 0.0, "B": 0.0, "M": 0.0, "E": 0.0}
        }
        states = "".join(self._states)
        counter = Counter(states)
        for i in range(len(states)):
            if i+1 == len(states):
                continue
            trans_counts[states[i]][states[i+1]] += 1.0
        trans_state = {k: {kk: (vv+1)/counter[k] for kk, vv in v.items()} for k, v in trans_counts.items()}

        return trans_state

    def _cal_emit_state(self):
        """计算观测概率。"""
        word_dict = {word: 0.0 for word in "".join(self._vocab)}
        emit_counts = {
            "S": dict(word_dict),
            "B": dict(word_dict),
            "M": dict(word_dict),
            "E": dict(word_dict)
        }
        states = "".join(self._states)
        counter = Counter(states)
        for index in range(len(self._states)):
            for i in range(len(self._states[index])):
                emit_counts[self._states[index][i]][self._words[index][i]] += 1
        emit_state = {k: {kk: (vv+1)/counter[k] for kk, vv in v.items()} for k, v in emit_counts.items()}

        return emit_state

    def _process_content(self, lines):
        """处理句子中的符号。"""
        return ["".join([word for word in line.strip() if not self._is_puns(word)]) for line in lines]

    def _get_test_corpus(self, name):
        """获取测试语料。"""
        if name == "test":
            path = self.config.get("test_corpus_path")
        elif name == "test_gold":
            path = self.config.get("test_corpus_gold_path")
        else:
            raise ValueError("test or test_gold")

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        corpus = self._process_content(lines)

        return corpus

    @staticmethod
    def _stats(cut_corpus, gold_corpus):
        """计算准确率、召回率、F1值。"""
        success_count, cut_count, gold_count = 0, 0, 0
        for index in range(len(cut_corpus)):
            cut_sentence = cut_corpus[index].split(" ")
            gold_sentence = gold_corpus[index].split(" ")
            cut_count += len(cut_sentence)
            gold_count += len(gold_sentence)
            for word in cut_sentence:
                if word in gold_sentence:
                    success_count += 1
        recall = float(success_count) / float(gold_count)
        precision = float(success_count) / float(cut_count)
        f1 = (2 * recall * precision) / (recall + precision)
        return precision, recall, f1

    def _save_state(self, init_state, trans_state, emit_state):
        """保存状态概率。"""
        init_state_path = self.config.get("init_state_path")
        trans_state_path = self.config.get("trans_state_path")
        emit_state_path = self.config.get("emit_state_path")
        self._save_state_to_file(init_state, init_state_path)
        self._save_state_to_file(trans_state, trans_state_path)
        self._save_state_to_file(emit_state, emit_state_path)

    def _get_state(self, name):
        """获取状态概率。"""
        if name == "init":
            state_path = self.config.get("init_state_path")
        elif name == "trans":
            state_path = self.config.get("trans_state_path")
        elif name == "emit":
            state_path = self.config.get("emit_state_path")
        else:
            raise ValueError('state name must in ["init", "trans", "emit"].')
        state = self._read_state_from_file(state_path)

        return state

    def _get_init_state(self):
        """获取初始概率，转为HMM模型接受的数据形式。"""
        states = ["S", "B", "M", "E"]
        init_state = self._get_state("init")
        init_p = np.array([init_state[s] for s in states])

        return states, init_p

    def _get_trans_state(self):
        """获取转移概率，转为HMM模型接受的数据形式。"""
        trans_state = self._get_state("trans")
        trans_p = np.array(
            [[trans_state[s][ss] for ss in self.states] for s in self.states])

        return trans_p

    def _get_emit_state(self):
        """获取观测概率，转为HMM模型接受的数据形式。"""
        emit_state = self._get_state("emit")
        vocabs = []
        for s in self.states:
            vocabs.extend([k for k, v in emit_state[s].items()])
        vocabs = list(set(vocabs))
        emit_p = np.array(
            [[emit_state[s][w] for w in vocabs] for s in self.states])

        return vocabs, emit_p

    def _pre_process(self, word):
        """未知字处理。"""
        return (self.vocabs.index(word) if word in self.vocabs else
                len(self.vocabs) - 1)

    def _viterbi_decode(self, x):
        """维特比解码。"""
        # 将概率值转换为对数值
        init = np.log(self.init_p)              # 初始状态概率向量
        transition = np.log(self.trans_p)       # 状态转移概率矩阵
        emission = np.log(self.emit_p)[:, x]    # 观测概率矩阵（发射矩阵）

        # 初始化
        rows, cols = emission.shape
        paths = np.zeros((rows, cols))      # 用来保存每个观测值到每个状态的最大概率
        viterbi = np.zeros((rows, cols))    # 用来保存这个最大概率是从上一个观测值的哪个状态过来的
        viterbi[:, 0] = init + emission[:, 0]

        # 维特比计算
        for j in range(1, cols):
            for i in range(rows):
                prob = viterbi[:, j - 1] + transition[:, i] + emission[i, j]
                sort = np.argsort(prob)
                paths[i, j] = sort[-1]
                viterbi[i, j] = max(prob)

        # 预测的状态序列（逆序）
        state_sequence = np.empty(cols, dtype=int)
        last = int(np.argsort(viterbi[:, -1])[-1])  # 最后一个观测值的概率最大的行序
        state_sequence[0] = last

        # 状态序列最大的对数概率
        logprob = -viterbi[state_sequence[0], -1]

        for j in range(cols - 1):
            last = int(paths[last, cols - 1 - j])  # 上一个观测值的行序
            state_sequence[j + 1] = last

        return logprob, state_sequence[::-1]

    def cal_state(self):
        """计算三类状态概率。"""
        for word in self._words:
            self._word_to_states(word)
        init_state = self._cal_init_state()
        trans_state = self._cal_trans_state()
        emit_state = self._cal_emit_state()
        self._save_state(init_state, trans_state, emit_state)

    def test(self):
        """分词测试。"""
        test_corpus = self._get_test_corpus("test")
        gold_corpus = [
            sentence.replace("  ", " ").strip()
            for sentence in self._get_test_corpus('test_gold') if sentence
        ]
        cut_corpus = [
            self.cut(sentence).strip() for sentence in test_corpus if sentence]
        result = self._stats(cut_corpus, gold_corpus)
        return result

    def cut(self, sentence):
        """分词。"""
        x = np.array([self._pre_process(w) for w in sentence])   # 文本转id
        logprob, states = self._viterbi_decode(x)                # 维特比解码
        tags = np.array([self.states[t] for t in states])       # id转标签

        # 根据标签进行分词
        cut_sentence = ""
        for i in range(len(tags)):
            if tags[i] in ("S", "E"):
                cut_sentence += sentence[i] + " "
            else:
                cut_sentence += sentence[i]

        return cut_sentence
