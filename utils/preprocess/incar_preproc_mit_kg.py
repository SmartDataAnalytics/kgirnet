from nltk.corpus import stopwords
import string
# import torch
import json
import os
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
import numpy as np
# from utils.utils_graph import gen_adjacency_mat, get_degree_matrix, getER_vec
from collections import defaultdict
# from utils.preprocessor import get_fuzzy_match, fuzz
# from unidecode import unidecode
# from collections import OrderedDict
from utils.args import get_args
import os


class Preprocessor:
    '''
    Preprocessor class
    '''
    # def __init__(self, data_path='/home/debanjan/submission_soccer/data/incar/', vec_dim=300,
    def __init__(self, data_path='data/incar/', vec_dim=300,
                 # fasttext_model='/home/debanjan/acl_submissions/soccerbot_acl/vocab/wiki.simple.bin'):
                 fasttext_model='/home/deep/Emphatic_VW/emotion_classifer-cnn/vectors/wiki.en.bin'):
        self.data_path = data_path
        # self.max_similarity = 85
        self.vec_dim = vec_dim

        cap_path = datapath(fasttext_model)
        self.word_emb = load_facebook_model(cap_path)
        # print (self.max_er_vec)
        self.stop = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.er_dict, self.global_ent, self.eo_dict = self.get_kg(self.data_path + 'KG/')

        self.args = get_args()
        self.train_dataset = self.get_data('train')
        self.val_dataset = self.get_data('val')
        self.test_dataset = self.get_data('test')

        self.entitites = [d['e'] for d in self.train_dataset]
        self.entitites = list(set(self.entitites))
        #  Create the vocab
        self.vocab = defaultdict(float)

        # self.vocab[pos]
        self.get_vocab(self.train_dataset)
        self.get_vocab(self.test_dataset)
        self.get_vocab(self.val_dataset)

        # Add additional tokens to vocab
        self.vocab[self.args.unk_tok] += 1.0
        self.vocab[self.args.sos_tok] += 1.0
        # self.vocab[self.args.ent_tok] += 1.0
        if not self.args.use_bert:
            self.vocab[self.args.eou_tok] += 1.0
        self.vocab[self.args.eos_tok] += 1.0

        self.stoi = dict(zip(self.vocab.keys(), range(1, len(self.vocab)+1)))
        self.stoi[self.args.pad_tok] = 0

        self.itos = {v: k for k, v in self.stoi.items()}
        print (len(self.stoi))
        self.n_words = len(self.stoi)

        self.vectors = np.zeros((len(self.stoi), vec_dim))
        for w, w2i in self.stoi.items():
            # if w2i < self.stoi[self.args.eos_tok]:
            self.vectors[w2i] = self.get_w2v(w)
        self.ent_dict = dict(zip(list(self.entitites), range(0, len(self.entitites))))
        # self.ent_dict[self.args.no_ent_tok] = 0

    def get_vocab(self, dataset):
        '''
        Get string to index for datasets
        :param dataset:
        :return:
        '''

        for conv in dataset:
            for k, v in conv.items():
                if k == 'a' or k == 'q':
                    # print (v)
                    sents = v
                    for w in sents.split():
                        # if w not in self.objects:
                            self.vocab[w] += 1.0

    def get_w2v(self, word):
        # get word2vecs
        if '_' not in word:
            return self.word_emb.wv[word]
        else:
            return self.get_avg_word2vec(word.split('_'))

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300)
        if phrase=="":
            return vec
        # phrase = phrase.strip()  # Remove _ during similarity calculations
        for w in phrase:
            # if w not in self.stop and w not in self.punc:
            vec = vec + self.word_emb.wv[w]
        return vec / float(len(phrase))

    def get_conv(self, filename):
        '''
        load all the training data
        :param filename:
        :return:
        '''
        with open(filename, 'r') as inp:
            try:
                out_f = json.load(inp)
            except Exception:
                print (filename)
        return out_f

    def load_all_files(self, dataset):
        '''
        Get list of all files in the directory
        :param dataset:
        :return:
        '''
        dat_files = os.listdir(self.data_path + '/manually_annotated/' + dataset + '_sketch/')
        return dat_files

    @staticmethod
    def get_kg(kg_path):
        '''
        Get the KG dictionary
        :param file_name:
        :return:
        '''
        kg_files = os.listdir(kg_path)
        er_pair_dict = defaultdict(set)
        eo_pair_dict = defaultdict(set)
        ent_l = set()
        for kgf in kg_files:
            with open(kg_path + kgf, 'r') as f:
                kg = f.readlines()
            for t in kg:
                e, r, o = t.replace('\n', '').split('\t')
                er_pair_dict[e+'#'+r].add(o)
                eo_pair_dict[e+'#'+o].add(r)
                ent_l.add(e)
                if 'weather' not in kgf:
                    ent_l.add(o)
                    er_pair_dict[o + '#' + r].add(e)
                    eo_pair_dict[e + '#' + o].add(r)
        return er_pair_dict, ent_l, eo_pair_dict

    def get_data(self, dataset):
        '''
        Load conversations from files
        :param dataset:
        :return:
        '''
        data_out = []
        # found = []
        data_files = self.load_all_files(dataset)
        for dat in data_files:
            # print (dat)
            # print ("dat is here")
            convo = self.get_conv(self.data_path + 'manually_annotated/' + dataset + '_sketch/' + dat)  # Get Conversations

            conv_hist = []
            for j, c in enumerate(convo):
                convo_dict = {}
                convo_dict['f'] = dat.replace('.json', '')
                # conv_hist = []
                if conv_hist:
                    try:
                        conv_hist.append(c['q' + str(j + 1)])
                    except Exception as e:
                        print (dat, e)
                    if self.args.use_bert:
                        convo_dict['q'] = (' '+self.args.bert_sep+' ').join(u for u in conv_hist)  # For bert
                    else:
                        convo_dict['q'] = (' '+self.args.eou_tok+' ').join(u for u in conv_hist)
                    try:
                        conv_hist.append(c['a' + str(j + 1)])
                    except Exception as e:
                        print (e, dat)
                        exit()
                else:
                    convo_dict['q'] = c['q' + str(j + 1)]
                    conv_hist.append(c['q' + str(j + 1)])
                    conv_hist.append(c['a' + str(j + 1)])
                convo_dict['q'] = convo_dict['q']
                convo_dict['_a'] = c['a' + str(j + 1)]
                convo_dict['_q'] = c['q' + str(j + 1)]
                # convo_dict['a'] = c['a' + str(j + 1)]

                # Get KG

                if c['input_ent' + str(j + 1)]:
                    convo_dict['e'] = c['input_ent' + str(j + 1)]
                    if 'changs' in convo_dict['e']:
                        convo_dict['e'] = 'p_._f_._changs'
                else:
                    convo_dict['e'] = self.args.no_ent_tok

                convo_dict['o'] = c['obj' + str(j + 1)].split(',')
                convo_dict['r'] = c['corr_rel' + str(j + 1)].split(',')
                convo_dict['kvr'] = c['kvr_entlist_qa' + str(j + 1)]

                ir = []
                if 'navigate' in dat:  # add 2 hop information in navigation
                    with open(self.data_path + 'KG/' + dat.replace('.json', '') + '_kg.txt', 'r') as f:  # Read KG file
                        nav_kg = f.readlines()
                    nav_dict = defaultdict(list)
                    nav_1hop = defaultdict(list)
                    for t in nav_kg:
                        e, r, o = t.replace('\n', '').split('\t')
                        nav_dict[e].append([r, o])
                        nav_dict[o].append([r, e])
                        # nav_1hop[e].append(o)
                        # nav_1hop[o].append(e)
                    if convo_dict['e'] in nav_dict.keys():
                        for w in convo_dict['_a'].split():
                            objects = [o for r, o in nav_dict[convo_dict['e']]]
                            _2hop_obj = []
                            for o in objects:
                                for conn_obj in nav_dict[o]:
                                    _2hop_obj.append(conn_obj)
                            if convo_dict['e'] in nav_dict.keys():  # if object is in 1 hop
                                if w == convo_dict['e']:
                                    ir.append('@entity')
                                elif w in objects:
                                    relation = [r for r, o in nav_dict[convo_dict['e']] if o == w]
                                    # object = w
                                    ir.append('@'+relation[0])
                                elif w in [o for r, o in _2hop_obj]:
                                    # _2hop_obj = [o for r, o in nav_dict[object]]
                                    # if w in _2hop_obj:
                                    relation = [r for r, o in _2hop_obj if o == w]
                                        #object = w
                                    ir.append('@@' + relation[0])
                                    #else:
                                    #    ir.append(w)
                                else:
                                    ir.append(w)

                elif 'weather' not in dat: # Process data for weather separately
                    for w in convo_dict['_a'].split():
                        if '' not in convo_dict['o'] and 'none' not in convo_dict['o']:
                            if w in convo_dict['o']:  # If word in output objects
                                ans_obj = [o for o in convo_dict['o'] if o == w]
                                ans_obj = ans_obj[0]
                                if convo_dict['e']: # check if there's input entity
                                    if convo_dict['e'] not in self.global_ent:
                                        print (convo_dict['e'], dat)
                                    try:
                                        # connected_reln = self.eo_dict[convo_dict['e']+'#'+ans_obj[0]]
                                        connected_reln = [r for r in convo_dict['r'] if r
                                                          in self.eo_dict[convo_dict['e']+'#'+ans_obj]]
                                        ir.append('@' + connected_reln[0])
                                    except Exception as e:
                                        # print (e)
                                        # print (dat)
                                        ir.append(w)
                                        # exit()
                                else:
                                    ir.append(w)
                            elif w == convo_dict['e']:
                                ir.append('@entity')
                            else:
                                ir.append(w)
                        else:
                            ir.append(w)
                else:
                    with open(self.data_path + 'KG/' + dat.replace('.json', '') + '_kg.txt', 'r') as f:  # Read KG file
                        weather_kg = f.readlines()
                    weather_dict = defaultdict(list)
                    for t in weather_kg:
                        e, r, o = t.replace('\n', '').split('\t')
                        weather_dict[e].append([r, o])
                    if convo_dict['e'] in weather_dict.keys():
                        for w in convo_dict['_a'].split():
                            objects = [o for r, o in weather_dict[convo_dict['e']]]
                            if convo_dict['e'] in weather_dict.keys():
                                if w == convo_dict['e']:
                                    ir.append('@entity')
                                elif w in objects:
                                    relation = [r for r, o in weather_dict[convo_dict['e']] if o == w]
                                    ir.append('@'+relation[0])
                                else:
                                    ir.append(w)

                convo_dict['a'] = ' '.join(ir)
                # if answer is empty put the original answer
                if not convo_dict['a']:
                    convo_dict['a'] = convo_dict['_a']
                data_out.append(convo_dict)

        return data_out


if __name__ == '__main__':
    preproc = Preprocessor()
    # file names
    if not preproc.args.use_bert:
        preproc_dir = preproc.data_path + 'preproc_files/'
    else:
        preproc_dir = preproc.data_path + 'preproc_files_bert/'
    train_preproc_file = preproc_dir + '/train.npy'
    valid_preproc_file = preproc_dir + '/val.npy'
    test_preproc_file = preproc_dir + '/test.npy'
    stoi_f = preproc_dir + '/stoi.npy'
    entity_dict_f = preproc_dir + '/etoi.npy'
    w2emb_f = preproc_dir + '/wemb.npy'

    # Save all files
    np.save(train_preproc_file, preproc.train_dataset)
    np.save(valid_preproc_file, preproc.val_dataset)
    np.save(test_preproc_file, preproc.test_dataset)
    np.save(stoi_f, preproc.stoi)
    np.save(entity_dict_f, preproc.ent_dict)
    np.save(w2emb_f, preproc.vectors)
