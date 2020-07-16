from unidecode import unidecode
import re
import numpy as np
from fuzzywuzzy import process, fuzz
from spacy.lang.en.stop_words import STOP_WORDS


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
    return string.strip().lower()


def generate_ngrams(s, n=[1, 2, 3, 4]):
    words_list = s.split()
    words_list = [w for w in words_list if w not in STOP_WORDS]
    ngrams_list = []

    for num in range(0, len(words_list)):
        for l in n:
            ngram = ' '.join(words_list[num:num + l])
            ngrams_list.append(ngram)
    return ngrams_list


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


# print (get_fuzzy_match('lionel messi', 'messi is the best striker'))