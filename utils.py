from functools import partial

import numpy as np

MAX_WORD_LEN = 16
MAX_SENT_LEN = 30

# russian letters and basic punctuation
index2let = [chr(i) for i in range(1072, 1104)] + list(".,!?")
let2index = {let: i + 1 for i, let in enumerate(index2let)}
index2let = set(index2let)

LETTERS_COUNT = len(index2let)


def preprocess_word(word):
    word = [let2index[ch] for ch in word if ch in index2let]
    if len(word) <= MAX_WORD_LEN:
        word += [0 for _ in range(MAX_WORD_LEN - len(word))]
        return word
    return None


def preprocess(sent):
    sent = sent.lower()
    sent = ''.join([i if i in index2let else ' ' for i in sent])
    for i in ".,!?":
        sent = sent.replace(i, " {} ".format(i))
    sent = sent.split()
    sent = [preprocess_word(word) for word in sent]
    sent = [s for s in sent if s]
    sent += [[0 for _ in range(MAX_WORD_LEN)]
             for _ in range(MAX_SENT_LEN - len(sent))]
    return np.asarray(sent)


def read_dataset(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def permute_dataset(dataset, labels):
    ids = np.random.permutation(labels.shape[0])
    return dataset[ids], labels[ids]


def prepare_train_data(pos_file, neg_file):
    pos_dataset = read_dataset(pos_file)
    neg_dataset = read_dataset(neg_file)
    full_dataset = pos_dataset + neg_dataset

    labels = np.zeros((len(full_dataset),), dtype=np.float32)
    labels[range(len(pos_dataset)),] = 1.

    preprocessed_lines = [preprocess(line) for line in full_dataset]
    preprocessed_lines = np.asarray(preprocessed_lines)

    return permute_dataset(preprocessed_lines, labels)
