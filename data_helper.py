import itertools
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
import pickle

logging.getLogger().setLevel(logging.INFO)


def clean_str(s):
    s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)

    # get only 1 repeated pattern (concatenation of PVs)
    # http://stackoverflow.com/questions/8672853/detecting-a-repeating-cycle-in-a-sequence-of-numbers-python
    try:
        regex = re.compile(r"(.+ .+)( \1)+")
        match = regex.search(s)
        s = match.group(1)
    except:
        return s
    return s.strip().lower()


def load_embeddings(vocabulary, dim=300):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, dim)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None: # Train
        sequence_length = max(len(x) for x in sentences)
    else: # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data(filename):
    df = pd.read_json(filename, lines=True)
    selected = ['indexedLabel', 'cleanText']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))

    labels = sorted(list(set(df[selected[0]].tolist())))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    print 'cleaning text, x_raw, y_raw'
    x_raw = df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    print 'cleaned text, x_raw, y_raw'

    print 'padding sentences'
    x_raw = pad_sentences(x_raw)
    print 'pad sentences done'

    print 'building vocabulary'
    vocabulary, vocabulary_inv = build_vocab(x_raw)
    print 'vocabulary built'

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)

    pickle.dump(x, open("./data/x.p", "wb"))
    pickle.dump(y, open("./data/y.p", "wb"))
    pickle.dump(vocabulary, open("./data/vocabulary.p", "wb"))
    pickle.dump(vocabulary_inv, open("./data/vocabulary_inv.p", "wb"))
    pickle.dump(df, open("./data/df.p", "wb"))
    pickle.dump(labels, open("./data/labels.p", "wb"))

    return x, y, vocabulary, vocabulary_inv, df, labels

if __name__ == "__main__":
    train_file = '../RNN_text_classification/data_tf.json'
    load_data(train_file)
