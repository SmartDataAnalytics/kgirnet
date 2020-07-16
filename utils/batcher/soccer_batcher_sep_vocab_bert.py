import torch
import numpy as np
import random
# from utils.utils_graph import gen_adjacency_mat, get_degree_matrix, getER_vec
# from collections import defaultdict
# import pandas as pd
# from utils.preprocessor import get_fuzzy_match, fuzz
from unidecode import unidecode
from utils.args import get_args
from pytorch_pretrained_bert import BertTokenizer
import os
from collections import defaultdict
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
# import torch.nn as nn


class SoccerBatcher:
    '''
    Class for handling soccer batches
    '''
    def __init__(self, data_path='/home/debanjan/submission_soccer/data/soccer/',
                 pretrained_weights='bert-base-uncased', use_bert=True, min_vocab_freq=1.0,
                 fasttext_model='/home/debanjan/acl_submissions/soccerbot_acl/vocab/wiki.simple.bin',
                 batch_size=32, max_sent_len=20, vec_dim=300, max_resp_len=15, gpu=False, domain='soccer'):
                 # fasttext_model='/home/debanjan/acl_submissions/soccerbot_acl/vocab/wiki.simple.bin',
        self.args = get_args()
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.max_out_len = max_resp_len
        self.vec_dim = vec_dim
        self.gpu = gpu
        self.n_graph_features = 1
        cap_path = datapath(fasttext_model)
        self.word_emb = load_facebook_model(cap_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # SRC and TRG vocabularies
        self.src_vocab = defaultdict(float)
        self.trg_vocab = defaultdict(float)
        # Load Datasets and preprocess files
        self.train_dataset = np.load(self.data_path+'preproc_files_kg/'+'train.npy', allow_pickle=True)
        # random.shuffle(self.train_dataset)
        self.val_dataset = np.load(self.data_path+'preproc_files_kg/'+'val.npy', allow_pickle=True)
        self.test_dataset = np.load(self.data_path+'preproc_files_kg/'+'test.npy', allow_pickle=True)
        self.conv2kg_mapping = np.load(self.data_path+'convfile2kg_mapping.npy', allow_pickle=True).item()
        # Create vocabularies
        self.create_vocab(self.train_dataset)
        self.create_vocab(self.val_dataset)
        self.create_vocab(self.test_dataset)
        self.trg_vocab[self.args.sos_tok] = len(self.train_dataset)  # setting this equal to train dataset length
        self.trg_vocab[self.args.eos_tok] = len(self.train_dataset)
        self.trg_vocab[self.args.unk_tok] = 10.0
        # remove words less than minimum_vocab_freq in target
        if domain == 'soccer':  # only do it for soccer
            self.trg_vocab = {k: v for k, v in self.trg_vocab.items() if v > min_vocab_freq}
        self.src_stoi = dict(zip(self.src_vocab.keys(), range(0, len(self.src_vocab.keys()))))
        self.src_itos = {v: k for k, v in self.src_stoi.items()}
        self.trg_stoi = dict(zip(self.trg_vocab.keys(), range(0, len(self.trg_vocab.keys()))))
        self.trg_itos = {v: k for k, v in self.trg_stoi.items()}

        # self.stoi = np.load(self.data_path+'preproc_files_kg/'+'stoi.npy', allow_pickle=True).item()
        self.etoi = np.load(self.data_path+'preproc_files_kg/'+'etoi.npy', allow_pickle=True).item()
        # self.vectors = np.load(self.data_path+'preproc_files_kg/'+'wemb.npy', allow_pickle=True)
        # Remove vectors which are not present in source stoi
        self.src_vectors = np.zeros((len(self.src_stoi), self.vec_dim))
        self.trg_vectors = np.zeros((len(self.trg_stoi), self.vec_dim))
        # for w, i in self.src_stoi.items():
        #     self.src_vectors[i] = self.get_w2v(w)
        # for w, i in self.trg_stoi.items():
        #     self.trg_vectors[i] = self.get_w2v(w)
        # self.itos = {v: k for k, v in self.stoi.items()}
        self.itoe = {v: k for k, v in self.etoi.items()}

        self.er_dict, self.global_ent, self.eo_dict, self.e_o_1hop, self.e_r_l = self.get_kg(data_path+'KG/', dat=domain)

        # Maximum graph input feature
        # self.max_er_vec = []  # max er vector combination size
        # for dat in self.train_dataset:
        #     self.max_er_vec.append(sum(len(v) for k, v in dat['kgER'].items()))
        # self.max_out_reln = np.max(self.max_er_vec)
        # Data Statistics

        # self.n_words = len(self.stoi)
        self.n_train = len(self.train_dataset)
        self.n_val = len(self.val_dataset)
        self.n_test = len(self.test_dataset)

        # self.vectors = np.zeros((len(self.itos) + 1, vec_dim))

    def get_w2i(self, word, vocab):
        # try:
        return vocab[word]
        # except KeyError:
        # print (word)
        # return vocab[self.args.unk_tok]

    def get_i2w(self, idx):
        # print (idx)
        try:
            return self.trg_itos[idx]
        except KeyError:
            # print (idx)
            return self.args.unk_tok

    # @classmethod
    def create_vocab(self, dataset):
        for d in dataset:
            for w in d['q'].split():
                self.src_vocab[w] += 1.0
            for w in d['_a'].split():
                self.trg_vocab[w] += 1.0

    @staticmethod
    def get_kg(kg_path, dat='soccer'):
        '''
        Get the KG dictionary
        :param file_name:
        :return:
        '''
        kg_files = os.listdir(kg_path)
        er_pair_dict = defaultdict(set)
        eo_pair_dict = defaultdict(set)
        e_o_connection = defaultdict(set)
        ent_l = set()
        ent_r_l = defaultdict(set)
        for kgf in kg_files:
            if dat == 'soccer':  # get for soccer
                for kf in os.listdir(kg_path+kgf):
                    with open(kg_path+kgf+'/'+kf, 'r') as f:
                        kg = f.readlines()
                    for t in kg:
                        e, r, o = unidecode(t).lower().replace('\n', '').split('\t')
                        er_pair_dict[e+'#'+r].add(o)
                        eo_pair_dict[e+'#'+o].add(r)
                        ent_l.add(e)
                        e_o_connection[e].add(o)
                        ent_r_l[e].add(r)
                        ent_r_l[o].add(r)
                        eo_pair_dict[e + '#' + o].add(r)
            else:
                with open(kg_path + kgf, 'r') as f:
                    kg = f.readlines()
                for t in kg:
                    e, r, o = t.replace('\n', '').split('\t')
                    er_pair_dict[e + '#' + r].add(o)
                    eo_pair_dict[e + '#' + o].add(r)
                    ent_l.add(e)
                    ent_r_l[e].add(r)
                    ent_r_l[o].add(r)
                    e_o_connection[e].add(o)
                    # e_o_connection[o].add(e)
                    if 'weather' not in kgf:
                        ent_l.add(o)
                        ent_r_l[e].add(r)
                        er_pair_dict[o + '#' + r].add(e)
                    eo_pair_dict[e + '#' + o].add(r)
        return er_pair_dict, ent_l, eo_pair_dict, e_o_connection, ent_r_l

    def get_w2v(self, word):
        # get word2vecs
        # return torch.from_numpy(self.word_emb.wv[word])
        return self.word_emb.wv[word]

    def get_etoi(self, entity):
        try:
            return self.etoi[entity]
        except KeyError:
            return self.etoi[self.e_o_1hop[entity]]

    def get_iter(self, dataset, domain='soccer'):
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
                q, q_m, t_t, i_g, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg = self._load_batches(batch_data, dat=domain)
            else:
                q, q_m, t_t, i_g, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg = self._load_batches(batch_data, dat=domain)

            yield q, q_m, t_t, i_g, y, y_m, i_e, t_y, c_e, i_q, o_r, c_o, l_kg

    def _load_batches(self, batch_data, dat='soccer'):
        '''The main data batcher'''
        b_s = min(self.batch_size, len(batch_data))
        # print (b_s)
        max_len_q = np.max([len(ques['q'].split()) for ques in batch_data])
        max_len_q = max_len_q if max_len_q < self.max_sent_len else self.max_sent_len
        max_len_a = np.max([len(ans['_a'].split()) for ans in batch_data])
        max_len_a = max_len_a if max_len_a < self.max_out_len else self.max_out_len
        # inp_graph_max_size = np.max([len(getER_vec(kg['kgER'])) for kg in batch_data])
        # print (inp_graph_max_size)
        correct_er_pair = ''
        q = torch.zeros(b_s, max_len_q)
        # q = torch.zeros(b_s, max_len_q)
        q_m = torch.zeros(b_s, max_len_q)
        token_type_id = torch.zeros(b_s, max_len_q)
        y = torch.zeros(b_s, max_len_a)
        y_m = torch.zeros(b_s, max_len_a)
        i_e = torch.zeros(b_s)

        i_g = torch.ones(b_s, len(self.trg_stoi))
        # graph_ele = [self.trg_stoi[g] for g in self.trg_stoi.keys() if '@' in g or g in self.e_r_l]
        # i_g[:, graph_ele] = 0.0
        # graph_ele_np = i_g.numpy()
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
            if dat == 'soccer':
                correct_obj.append(data['o'])
            else:
                correct_obj.append(data['kvr'])
            local_kg.append(data['f'])
            if data['e']:
                if data['o'][0]:
                    correct_ent.append(data['e'])
                else:
                    correct_ent.append(self.conv2kg_mapping[data['f']].lower())
            else:
                correct_ent.append(self.args.no_ent_tok)
            if '' not in data['r']:
                relations.append('')
            # Get true responses
            true_answer.append(data['_a'])
            inp_query.append(data['_q'])
            # query = data['q'].split()[-max_len_q:]  # Get max length from the last of the context
            query = self.tokenizer.tokenize(data['q'].replace(self.args.eou_tok, self.args.bert_sep))[-(max_len_q - 2):]  # Get max length from the last of the context -2 for BERT CLS and SEP
            query = [self.args.bert_cls]+query+[self.args.bert_sep]
            # tokenized_q = self.tokenizer.tokenize(query)
            token_ids = self.tokenizer.convert_tokens_to_ids(query)
            q[b, :len(token_ids)] = torch.tensor(token_ids)
                # q[b, k, :] = self.get_w2v(w)
            q_m[b, :len(token_ids)] = 1
            # Handle SEP tag with index = 102
            sep_tags = [j for j, t_i in enumerate(token_ids) if t_i == 102]
            try:
                token_type_id[b, sep_tags[-2]+1: len(token_ids)] = 1
            except IndexError:
                token_type_id[b, :len(token_ids)] = 1
            answer = data['a'].split()[:(max_len_a-1)] + [self.args.eos_tok]
            for k, w_a in enumerate(answer):
                # print(self.get_w2i(w_a))
                try:
                    y[b, k] = self.get_w2i(w_a, self.trg_stoi)
                except KeyError:
                    # print ('here', w_a)
                    # print (data['e'])
                    # print (answer, query)
                    try:
                        y[b, k] = self.get_w2i(self.eo_dict[data['e']+'#'+w_a], self.trg_stoi)
                    except Exception:
                        y[b, k] = self.trg_stoi[self.args.unk_tok]
            y_m[b, :len(answer)] = 1.0
            if data['e']:
                if 'none' in data['e']:

                     i_e[b] = 0
                else:
                    try:
                        if data['o'][0]:
                            i_e[b] = self.etoi[data['e']]
                        else:
                            i_e[b] = self.etoi[self.conv2kg_mapping[data['f']].lower()]
                    except KeyError:
                        i_e[b] = self.etoi[self.args.no_ent_tok]
            else:
                continue
            if data['h']:  # convert input graph to vector
                for k, v in data['h'].items():
                    try:
                        i_g[b][self.trg_stoi[k]] = v
                    except KeyError:
                        k = k.replace(' ', '_')
                        try:
                            i_g[b][self.trg_stoi[k]] = v
                        except KeyError:
                            continue
        if self.gpu:
            q, q_m, token_type_id, y, y_m, i_e, i_g = q.cuda(), q_m.cuda(), token_type_id.cuda(), y.cuda(), y_m.cuda(), i_e.cuda(), i_g.cuda()

        return q.long(), q_m.long(), token_type_id.long(), i_g, y.long(), y_m, i_e.long(), true_answer, correct_ent, inp_query, relations, correct_obj, local_kg


if __name__ == '__main__':
    batcher = SoccerBatcher()
    batches = batcher.get_iter('val', domain='soccer')
    for e, b in enumerate(batches):
        # print (b)
        q, q_m, t_t, i_g, y, y_m, i_e, t_e, c_e, i_q, o_r, c_o, l_kg = b
        for j, ques in enumerate(q):
            print (ques)
            print (t_t[j])
