import os
import re
import json
import spacy
import numpy as np
from functools import partial
from unidecode import unidecode
from spacy.tokenizer import Tokenizer
from fuzzywuzzy import process, fuzz
from multiprocessing import Pool, cpu_count
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict, defaultdict

STOP_WORDS.add('de_l_la_le_di')

#spacy tokenizers
nlp = spacy.load('en')
pos = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)

#load word2index files
stoi_soccer = np.load("vocab/w2i_soccer.npy", allow_pickle=True).item()
itos_soccer = {v:k for k,v in stoi_soccer.items()}

stoi_incar = np.load("vocab/w2i_soccer.npy", allow_pickle=True).item()
itos_incar = {v:k for k,v in stoi_incar.items()}


correct_pos = ['NOUN', 'PROPN', 'ADJ', 'NUM', 'VERB']
w_h_words = ['what', 'how', 'when', 'where', 'why', 'who']

vocab_soccer_wiki300 = np.load("vocab/vocab_soccer_wiki300.npy", allow_pickle=True).item()
vocab_incar_wiki300 = np.load("vocab/vocab_incar_wiki300.npy", allow_pickle=True).item()

team_kgs = {}
hit2team_maps = np.load("data/convfile2kg_mapping.npy", allow_pickle=True)
kg_club = "data/KG/clubs/"
kg_country = "data/KG/country/"
kg_incar = "data/KG/incar/"

output_dir = "processed_data/"


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
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\"", "", string)
    return ' ' +string.strip().lower()+ ' '

def get_max_kb():
    kg_cl = os.listdir(kg_club)
    for kg_c in kg_cl:
        if kg_c:
            team_kgs[kg_c.replace('.txt', '')] = read_kg(kg_club+kg_c)

    kg_na = os.listdir(kg_country)
    for kg_n in kg_na:
        if kg_n:
            team_kgs[kg_n.replace('.txt', '')] = read_kg(kg_country+kg_n)

    kg_car = os.listdir(kg_incar)
    for kg_in in kg_car:
        if kg_in:
            team_kgs[kg_in.replace('.txt', '')] = read_kg(kg_incar+kg_in)

    max_len = np.max([(len(a)) for a, b, c in team_kgs.values()])
    return max_len


def getw2id(dataset, word):
    try:
        return stoi_soccer[word] if dataset=="soccer" else stoi_incar[word]
    except KeyError:
        return stoi_soccer['UNK'] if dataset=="soccer" else stoi_incar['UNK']


def getsent2i(dataset, sent):
    out = []
    sent = sent.strip()
    tokens = tokenizer(sent)
    for t in tokens:
        t = t.text
        out.append(getw2id(dataset, t))
    return out


def generate_ngrams(s, n=[1, 2, 3, 4]):
    words_list = s.split()
    words_list = [w for w in words_list if w not in STOP_WORDS]
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in n:
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    return ngrams_list


def read_kg(file_n):
    with open(file_n, 'r', encoding='utf-8') as f:
        kg_info = f.readlines()
    kg_info = [unidecode(l) for l in kg_info]
    kg_sub = [info.replace('\n', '').split('\t')[0].strip().lower() for info in kg_info]
    kg_reln = [info.replace('\n', '').split('\t')[1].strip().lower() for info in kg_info]
    kg_obj = [info.replace('\n', '').split('\t')[-1].strip().lower() for info in kg_info]
    return kg_sub, kg_reln, kg_obj



def check_question(question, dataset):
    if dataset=="soccer":
        question = ' '.join([itos_soccer[idx] for idx in question])
    else:
        question = ' '.join([itos_incar[idx] for idx in question])
    if '?' in question:
        return True
    elif any(map(question.split()[0].__contains__, w_h_words)):
        return True
    else:
        return False


def get_avg_word2vec(phrase, dataset):
    """get word vectors for phrases"""
    vec = np.zeros(300)
    phrase = phrase.strip()
    for w in phrase.split():
        if dataset=="soccer":
            vec = vec + vocab_soccer_wiki300[w].reshape(1, 300).astype(np.float32)
        else:
            vec = vec + vocab_incar_wiki300[w].reshape(1, 300).astype(np.float32)
    return vec


def get_rel_sim(relation, question, dataset):
    """
    Get max cosine distance for relations
    :param relation:
    :param question:
    :return:
    """
    query_ngrams = generate_ngrams(question)
    query_ngrams_vec = [get_avg_word2vec(phr, dataset) for phr in query_ngrams]
    relation_ngram = get_avg_word2vec(relation, dataset)

    similarities = [cosine_similarity(relation_ngram, q)[0][0] for q in query_ngrams_vec]
    if similarities and np.max(similarities) > 0.5:
        return np.max(similarities)
    else:
        return 0.0


def get_fuzzy_match(object, answer, threshold=80):
    """get phrase with highest match in answer"""
    answer_phrase = generate_ngrams(answer)
    if answer_phrase:
        best_match = [fuzz.ratio(object, phr) for phr in answer_phrase]
        if np.max(best_match)>threshold:
            return np.max(best_match), answer_phrase[np.argmax(best_match)]
        else:
            return 0,''
    else:
        return 0, ''


def check_presence(answer, kb_key):
    """check probable presence"""
    answer, match = process.extract(kb_key, answer)[0]
    if match > 0.5:
        return match
    else:
        return 0.0


def get_chunks(query):
    chunks = np.zeros((len(query.split())))
    doc = pos(query)
    for e in doc.noun_chunks:
        chunks[e.start: e.end] = 1
    return chunks


def read_json(dataset,file_name):

    json_f = file_name.split('/')[-1].replace('.json', '')
    team = hit2team_maps[json_f]
    sub, reln, obj = team_kgs[team + '_kg']
    sub = [getsent2i(s) for s in sub]
    reln = [getsent2i(r) for r in reln]
    with open(file_name, 'r', encoding='utf-8') as fp:
        conv = json.load(fp, object_pairs_hook=OrderedDict)
    q, q_c, a = [], [], []
    for k, v in conv.items():
        if 'q' in k:
            if dataset=="soccer":
                q.append(getsent2i(clean_str(v).strip()))
                q_c.append(get_chunks(clean_str(v).strip()))
            else:
                q.append(getsent2i(v.strip()))
                q_c.append(get_chunks(v.strip()))
        else:
            if dataset=="soccer":
                a.append(clean_str(v))
            else:
                a.appen(v)

    params = [(ans, team, q[j]) for j, ans in enumerate(a)]
    answer_mask = [1, 0, 0, 0]
    return q, q_c, a, [],[], team+'_kg'
    #return q, q_c, answers_replaced, sub, reln, team + '_kg'


def add_correct_relation(dataset):
    """
    Add correct relation to the conversation files
    :return:
    """
    corrected_datadir = "data/corrected_data/"
    datatypes = ["val","test","train"]

    if dataset == "soccer":
        for data in datatypes:
            allfiles = os.listdir("data/soccer_conversations/"+data+"/_with_entities/")
    else:
        pass


def generate_processed_data(dataset):
    """
    :param dataset:
    :return:
    """
    datatypes = ["val","test","train"]

    for data in datatypes:
        in_f = 'data/' +dataset+"_conversations/"+ data+'_with_entities/'
        dialogue_f = os.listdir(in_f)
        data_dial = [read_json(dataset,in_f + d_f) for d_f in dialogue_f]
        np.save(output_dir+dataset+"_"+data+".npy", data_dial)


if __name__ == '__main__':
    max_kb_size = get_max_kb()
    print('Saving team KG')
    np.save(output_dir+'team_kg.npy', team_kgs)
    print(max_kb_size)
    print(team_kgs.keys())
    generate_processed_data('soccer')
    generate_processed_data('incar')


