import torch
import numpy as np
# from utils.utils_graph import gen_adjacency_mat, get_degree_matrix, getER_vec
# from collections import defaultdict
# import pandas as pd
# from utils.preprocessor import get_fuzzy_match, fuzz
# from unidecode import unidecode
from utils.args import get_args
import os
from collections import defaultdict
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
# import torch.nn as nn


class InCarBatcherEnt:
    '''
    Class for handling soccer batches
    '''
    def __init__(self, data_path='data/incar/',
                 fasttext_model=os.path.join(os.getcwd(),'data/wiki.simple.bin'),
                 batch_size=32, max_sent_len=20, vec_dim=300, max_resp_len=15, gpu=False):
        self.args = get_args()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.max_out_len = max_resp_len
        self.vec_dim = vec_dim
        self.gpu = gpu
        self.n_graph_features = 1
        cap_path = datapath(fasttext_model)
        # self.word_emb = load_facebook_model(cap_path)

        # Load Datasets and preprocess files
        self.train_dataset = np.load(self.data_path+'preproc_files/'+'train.npy', allow_pickle=True)
        self.val_dataset = np.load(self.data_path+'preproc_files/'+'val.npy', allow_pickle=True)
        self.test_dataset = np.load(self.data_path+'preproc_files/'+'test.npy', allow_pickle=True)

        self.stoi = np.load(self.data_path+'preproc_files/'+'stoi.npy', allow_pickle=True).item()
        self.etoi = np.load(self.data_path+'preproc_files/'+'etoi.npy', allow_pickle=True).item()
        self.vectors = np.load(self.data_path+'preproc_files/'+'wemb.npy', allow_pickle=True)
        self.itos = {v: k for k, v in self.stoi.items()}
        self.itoe = {v: k for k, v in self.etoi.items()}
        self.er_dict, self.ent_list, self.eo_dict = self.get_kg(data_path+'KG/')

        # Maximum graph input feature
        # self.max_er_vec = []  # max er vector combination size
        # for dat in self.train_dataset:
        #     self.max_er_vec.append(sum(len(v) for k, v in dat['kgER'].items()))
        # self.max_out_reln = np.max(self.max_er_vec)
        # Data Statistics

        self.n_words = len(self.stoi)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)

        # self.vectors = np.zeros((len(self.itos) + 1, vec_dim))

    def get_w2i(self, word):
        try:
            return self.stoi[word]
        except KeyError:
            return self.stoi[self.args.unk_tok]

    def get_i2w(self, idx):
        # print (idx)
        try:
            return self.itos[idx]
        except KeyError:
            print (idx)
            return self.args.unk_tok

    @staticmethod
    def get_kg(kg_path):
        '''
        Get the KG dictionary
        :param file_name:
        :return:
        '''
        kg_files = os.listdir(kg_path)
        er_pair_dict = defaultdict()
        eo_pair_dict = defaultdict(set)
        ent_l = set()
        for kgf in kg_files:
            with open(kg_path + kgf, 'r') as f:
                kg = f.readlines()
            for t in kg:
                e, r, o = t.replace('\n', '').split('\t')
                er_pair_dict[e+'#'+r] = o
                eo_pair_dict[e+'#'+o].add(r)
                ent_l.add(e)
                if 'weather' not in kgf:
                    ent_l.add(o)
                    er_pair_dict[o + '#' + r] = o
                    eo_pair_dict[e + '#' + o].add(r)
        return er_pair_dict, ent_l, eo_pair_dict

    def get_w2v(self, word):
        # get word2vecs
        return torch.from_numpy(self.word_emb.wv[word])

    def get_iter(self, dataset):
        """main batcher"""
        if dataset == 'train':
            data = self.train_dataset
        elif dataset == 'val':
            data = self.val_dataset
        elif dataset == 'test':
            data = self.test_dataset

        for j in range(0, len(data), self.batch_size):
            batch_data = data[j:j+self.batch_size]
            if len(batch_data) < self.batch_size:
                q, q_m, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg = self._load_batches(batch_data)
            else:
                q, q_m, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg = self._load_batches(batch_data)

            yield q, q_m, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg

    def _load_batches(self, batch_data):
        '''The main data batcher'''
        b_s = min(self.batch_size, len(batch_data))
        # print (b_s)
        max_len_q = np.max([len(ques['q'].split()) for ques in batch_data])
        max_len_q = max_len_q if max_len_q < self.max_sent_len else self.max_sent_len
        max_len_a = np.max([len(ans['a'].split()) for ans in batch_data])
        max_len_a = max_len_a if max_len_a < self.max_out_len else self.max_out_len
        # inp_graph_max_size = np.max([len(getER_vec(kg['kgER'])) for kg in batch_data])
        # print (inp_graph_max_size)
        correct_er_pair = ''
        q = torch.zeros(b_s, max_len_q)
        # q = torch.zeros(b_s, max_len_q)
        q_m = torch.zeros(b_s, max_len_q)
        y = torch.zeros(b_s, max_len_a)
        y_m = torch.zeros(b_s, max_len_a)
        i_e = torch.zeros(b_s)
        # out_kg = torch.zeros(b_s, self.max_out_reln)
        # out_kg_mask = torch.zeros(b_s, self.max_out_reln)

        relations = []
        true_answer = []
        inp_query = []
        correct_ent = []
        correct_obj = []
        local_kg = []
        # inp_graph_m = torch.zeros(b_s, self.max_out_reln)

        for b, data in enumerate(batch_data):
            if '' not in data['o']:
                correct_obj.append(data['o'])
            else:
                correct_obj.append('')
            local_kg.append(data['f'])
            if data['e']:
                correct_ent.append(data['e'])
            else:
                correct_ent.append(self.args.no_ent_tok)
            if '' not in data['r']:
                relations.append('')
            # Get true responses
            true_answer.append(data['_a'])
            inp_query.append(data['_q'])
            query = data['q'].split()[-max_len_q:]  # Get max length from the last of the context
            for k, w in enumerate(query):  # Get word embeddings
                q[b, k] = self.get_w2i(w)
                # q[b, k, :] = self.get_w2v(w)
            q_m[b, :len(query)] = 1
            answer = data['a'].split()[:(max_len_a-1)] + [self.args.eos_tok]
            for k, w_a in enumerate(answer):
                # print(self.get_w2i(w_a))
                try:
                    y[b, k] = self.get_w2i(w_a)
                except Exception:
                    print (w_a)
            y_m[b, :len(answer)] = 1.0
            if data['e']:
                if 'none' in data['e']:
                     i_e[b] = 0
                else:
                    try:
                        i_e[b] = self.etoi[data['e']]
                    except KeyError:
                        i_e[b] = self.etoi[self.args.no_ent_tok]
            else:
                continue
        if self.gpu:
            q, q_m, y, y_m, i_e = q.cuda(), q_m.cuda(), y.cuda(), y_m.cuda(), i_e.cuda()

        return q.long(), q_m, y.long(), y_m, i_e.long(), true_answer, correct_ent, inp_query, relations, correct_obj, local_kg


if __name__ == '__main__':
    batcher = InCarBatcher()
    batches = batcher.get_iter('val')
    for e, b in enumerate(batches):
        # print (b)
        q, q_m, y, y_m, i_e, t_e, c_e, i_q, o_r, c_o, l_kg = b
        print (c_o)

