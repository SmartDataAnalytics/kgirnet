import json
import re
import os
import spacy
from unidecode import unidecode
import numpy as np
from collections import defaultdict, OrderedDict
from spacy.tokenizer import Tokenizer
from gensim.models import KeyedVectors

# tokenizer
nlp = spacy.load('en')
tokenizer = Tokenizer(nlp.vocab)

#files and directories
vocab_incar_f = "vocab/vocab_incar.npy"
vocab_soccer_f = "vocab/vocab_soccer.npy"
vocab_soccer_wiki300_f = "vocab/vocab_soccer_wiki300.npy"
vocab_incar_wiki300_f  = "vocab/vocab_incar_wiki300.npy"
w2id_soccer_f = 'vocab/w2i_soccer.npy'
w2id_incar_f = 'vocab/w2i_incar.npy'

incar_train = "data/incar/conversations/train_with_entities_r/"
soccer_train = "data/soccer/conversations/train_with_entities_r/"

kg_club = "data/soccer/KG/clubs/"
kg_country = "data/soccer/KG/country/"
kg_incar = "data/soccer/KG/incar/"

min_freq = 1
pretrained_vector = "vocab/enwiki_20180420_300d.txt"
vector_dim = 300


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = unidecode(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\"", "", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\.", " ", string)
    return string.strip().lower()


def read_json(file_n):
    with open(file_n, 'r', encoding='utf-8') as fp:
        conv = json.load(fp, object_pairs_hook=OrderedDict)
    queries = []
    answers = []
    for k, v in conv.items():
        if 'q' in k:
            queries.append(v)
        else:
            answers.append(v)
    return queries, answers


def read_kg(file_n):
    """
    Get kg subject and relations
    :param file_n: input kg for team
    :return:
    """
    with open(file_n, 'r', encoding='utf-8') as f:
        kg_info = f.readlines()
    kg_sub_reln = [re.sub("\s\s+" , " ", ' '.join(info.replace('\n', '').split('\t')[0:2])) for info in kg_info]

    return kg_sub_reln


def create_w2i(train_path):
    all_sents = []
    train_files = os.listdir(train_path)
    for tr_f in train_files:
        ques, ans = read_json(train_path + tr_f)
        for q in ques:
            all_sents.append(unidecode(q))
        for a in ans:
            all_sents.append(unidecode(a))

    # Adding all KG tokens
    if "soccer" in train_path:      #for soccer KG
        all_kgs_club = os.listdir(kg_club)
        for kb in all_kgs_club:
            kg_cl = read_kg(kg_club + kb)
            for s_r in kg_cl:
                all_sents.append(unidecode(s_r))
        all_kgs_nation = os.listdir(kg_country)
        for kb in all_kgs_nation:
            kg_na = read_kg(kg_country + kb)
            for s_r in kg_na:
                all_sents.append(unidecode(s_r))
    else:                           #for inacar KG
        all_kgs_incar = os.listdir(kg_incar)
        for kb in all_kgs_incar:
            kg_cl = read_kg(kg_incar+kb)
            for s_r in kg_cl:
                all_sents.append(unidecode(s_r))

    all_sents = [clean_str(sent) for sent in all_sents]


    print('Creating the vocabulary.....')
    vocab = defaultdict(float)
    for sent in all_sents:
        tokens = tokenizer(sent)
        for token in tokens:
            token = token.text
            if token:
                if isinstance(token, str):
                    vocab[token.lower()] += 1.0
                else:
                    vocab[token] += 1.0
    print('Created vocab dictionary with length:' + str(len(vocab)))

    if "soccer" in train_path:
        np.save(vocab_soccer_f, vocab)
    else:
        np.save(vocab_incar_f,vocab)

    unq_w = []
    unq_w.append('PAD')
    for w, c in vocab.items():
        if c > min_freq:
            unq_w.append(w)

    word2id = dict(zip(unq_w, range(0, len(unq_w))))
    print('Total words in vocab: ' + str(len(word2id)))
    return vocab, word2id

"""
wv_from_text = KeyedVectors.load_word2vec_format(pretrained_vector, binary=False)
print(wv_from_text["rony"])
"""

if __name__ == "__main__":
    vocab_soccer, w2i_soccer = create_w2i(soccer_train)
    w2i_soccer['UNK'] = len(w2i_soccer)+1
    w2i_soccer['SOS'] = len(w2i_soccer)+1
    w2i_soccer['EOS'] = len(w2i_soccer)+1

    vocab_soccer_wiki300d_vector = {}
    print ('Loading wikipedia2vec vectors..........')
    wv_from_wiki = KeyedVectors.load_word2vec_format(pretrained_vector, binary=False)
    for word in w2i_soccer.keys():
        if word in wv_from_wiki.vocab:
            vocab_soccer_wiki300d_vector[word] = np.array(wv_from_wiki[word])
        else:
            vocab_soccer_wiki300d_vector[word] = np.zeros(vector_dim)

    for i in range(0, 200):
        w2i_soccer['o'+str(i)] = len(w2i_soccer) + 1

    np.save(w2id_soccer_f, w2i_soccer)
    np.save(vocab_soccer_wiki300_f, vocab_soccer_wiki300d_vector)

    #Do the samme as above                   FOR INCAR

    vocab_incar, w2i_incar = create_w2i(incar_train)
    w2i_incar['UNK'] = len(w2i_incar)+1
    w2i_incar['SOS'] = len(w2i_incar)+1
    w2i_incar['EOS'] = len(w2i_incar)+1

    vocab_incar_wiki300d_vector = {}
    for word in w2i_incar.keys():
        if word in wv_from_wiki.vocab:
            vocab_incar_wiki300d_vector[word] = np.array(wv_from_wiki[word])
        else:
            vocab_incar_wiki300d_vector[word] = np.zeros(vector_dim)

    for i in range(0, 200):
        w2i_incar['o'+str(i)] = len(w2i_incar) + 1

    np.save(w2id_incar_f, w2i_incar)
    np.save(vocab_incar_wiki300_f, vocab_incar_wiki300d_vector)
