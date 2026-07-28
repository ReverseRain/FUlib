import numpy as np
import os
import sys
import random
import torch
from collections import Counter
from datasets import load_dataset
from utils.dataset_utils import check, separate_data, split_data, save_file, sample_proxy

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

random.seed(1)
np.random.seed(1)
num_clients = 10
dir_path = "AGNews/"


def build_vocab_and_tokenize(texts, max_vocab_size=32000, max_len=200):
    """
    1. 构建全局词表 (Vocab)
    2. 将字符串文本转化为词索引序列 (Indices Sequence) 和实际长度 (Length)
    """
    print("正在构建词表并进行文本 Index 化...")
    # 统计词频
    counter = Counter()
    tokenized_texts = []

    for text in texts:
        tokens = text.lower().split()[:max_len]
        counter.update(tokens)
        tokenized_texts.append(tokens)

    # 留出 0 作为 <pad>, 1 作为 <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)

    processed_data = []
    for tokens in tokenized_texts:
        # 将单词转化为数字 ID
        indices = [vocab.get(token, 1) for token in tokens]
        length = len(indices)

        # 填充 (Padding) 到固定长度 max_len，保证 Tensor 可以对齐
        if length < max_len:
            indices = indices + [0] * (max_len - length)
        else:
            indices = indices[:max_len]
            length = max_len

        # 存入 (索引列表, 真实长度)
        processed_data.append((indices, max(1, length)))

    return processed_data


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train/")
    test_path = os.path.join(dir_path, "test/")
    proxy_path = os.path.join(dir_path, "proxy/")

    # 注意：重新打标签和处理数据时，需要覆盖生成
    # if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    #     return

    print("正在加载 AG News 数据集...")
    dataset = load_dataset("ag_news")

    raw_texts = []
    dataset_label = []

    for item in dataset['train']:
        dataset_label.append(item['label'])
        raw_texts.append(item['text'])

    for item in dataset['test']:
        dataset_label.append(item['label'])
        raw_texts.append(item['text'])

    # 将文本转化为 Index 化的 (indices, length) 格式
    processed_dataset_x = build_vocab_and_tokenize(raw_texts, max_vocab_size=32000, max_len=200)

    dataset_text = np.array(processed_dataset_x, dtype=object)
    dataset_label = np.array(dataset_label, dtype=np.int64)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data((dataset_text, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    proxy_data = sample_proxy(list(dataset_text))
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, proxy_path, proxy_data, niid, balance, partition)


if __name__ == "__main__":
    sys_argv1 = sys.argv[1] if len(sys.argv) > 1 else "noniid"
    sys_argv2 = sys.argv[2] if len(sys.argv) > 2 else "balance"
    sys_argv3 = sys.argv[3] if len(sys.argv) > 3 else "-"

    niid = True if sys_argv1 == "noniid" else False
    balance = True if sys_argv2 == "balance" else False
    partition = sys_argv3 if sys_argv3 != "-" else None

    dir_path = "AGNews/"

    generate_dataset(dir_path, num_clients, niid, balance, partition)