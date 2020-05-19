from hmm_segment import HMMSegment


def train(config):
    """语料处理及HMM模型概率计算。"""
    model = HMMSegment(config, mode="train")
    model.cal_state()


def test(config):
    """模型测试。"""
    model = HMMSegment(config)
    return model.test()


def cut(sentence, config):
    """分词。"""
    model = HMMSegment(config)
    return model.cut(sentence)


if __name__ == '__main__':
    config_dict = {
        'train_corpus_path': 'data/msr_training.txt',
        'test_corpus_path': 'data/msr_test.txt',
        'test_corpus_gold_path': 'data/msr_test_gold.txt',

        'init_state_path': 'states/init_state.pkl',
        'trans_state_path': 'states/trans_state.pkl',
        'emit_state_path': 'states/emit_state.pkl',
    }   # 配置

    train(config_dict)          # 训练
    print(test(config_dict))    # 测试
    print(cut("不少人遇到打击和挫折时，怨恨生活不公平。", config_dict))     # 分词
