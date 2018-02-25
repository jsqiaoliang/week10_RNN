#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import random

import numpy as np


def read_data(filename):
    with open(filename, encoding="utf-8") as f:
        data = f.read()
    data = list(data)
    return data


def index_data(sentences, dictionary):
    shape = sentences.shape
    sentences = sentences.reshape([-1])
    index = np.zeros_like(sentences, dtype=np.int32)
    for i in range(len(sentences)):
        try:
            index[i] = dictionary[sentences[i]]
        except KeyError:
            index[i] = dictionary['UNK']

    return index.reshape(shape)


def get_train_data(vocabulary, batch_size, num_steps):
    data, count, dictionary, reversed_dictionary = build_dataset(vocabulary, 5000)
    # chars = list(set(data))
    data_size = len(data)
    print('data Size size', data_size)
    data_partition_size = data_size // batch_size
    print('data partition size', data_partition_size)

    data_x = np.zeros([batch_size, data_partition_size], dtype=np.int32)
    data_y = np.zeros([batch_size, data_partition_size], dtype=np.int32)

    char_to_ix = {ch: i for i,ch in enumerate(data)}
    ix_to_char = {i: ch for i,ch in enumerate(data)}
    raw_x = [char_to_ix[ch] for ch in data]
    raw_y = [char_to_ix[ch] for ch in data[1:]]

    for i in range(batch_size):
        data_x[i] = raw_x[data_partition_size * i:data_partition_size * (i + 1)]
        data_y[i] = raw_y[data_partition_size * i:data_partition_size * (i + 1)]
    epoch_size  = data_partition_size // num_steps
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x,y)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
