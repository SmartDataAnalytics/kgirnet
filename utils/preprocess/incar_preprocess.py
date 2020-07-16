from nltk.corpus import stopwords
import string
import torch
import json
import os
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model
import numpy as np
from utils.utils_graph import gen_adjacency_mat, get_degree_matrix, getER_vec
from collections import defaultdict
# from utils.preprocessor import get_fuzzy_match, fuzz
from unidecode import unidecode
from collections import OrderedDict
from utils.args import get_args


class Preprocessor:
    '''
    Preprocessor class
    '''
    # def __init__(self, data_path='/home/debanjan/submission_soccer/data/incar/', vec_dim=300,
    def __init__(self, data_path='data/incar/', vec_dim=300,
                 # fasttext_model='/home/debanjan/acl_submissions/soccerbot_acl/vocab/wiki.simple.bin'):
                 fasttext_model='/data/dchaudhu/soccerbot_acl/vocab/wiki.en.bin'):
        self.data_path = data_path
        self.max_similarity = 85
        self.vec_dim = vec_dim
        cap_path = datapath(fasttext_model)
        self.word_emb = load_facebook_model(cap_path)
        # print (self.max_er_vec)
        self.stop = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.er_dict, self.global_ent = self.get_kg(self.data_path + 'KG/')
        self.args = get_args()
        self.train_dataset = self.get_data('train')
        self.val_dataset = self.get_data('val')
        self.test_dataset = self.get_data('test')
        self.max_er_vec = []  # max er vector combination size
        for dat in self.train_dataset:
            self.max_er_vec.append(sum(len(v) for k, v in dat['kgER'].items()))
        self.max_out_reln = np.max(self.max_er_vec)
        self.inp_graph_max_size = np.max([len(getER_vec(kg['kgER'])) for kg in self.train_dataset])
        print('input graph size:'+str(self.inp_graph_max_size))
        print(self.max_out_reln)
        self.objects = ['o'+str(j) for j in range(self.max_out_reln)]
        # Create vocabulary and word2id
        self.vocab = defaultdict(float)
        # self.vocab[pos]
        self.get_vocab(self.train_dataset)
        self.get_vocab(self.test_dataset)
        self.get_vocab(self.val_dataset)
        self.vocab[self.args.unk_tok] += 1.0
        self.vocab[self.args.sos_tok] += 1.0
        # self.vocab[self.args.noent_tok] += 1.0
        self.vocab[self.args.eou_tok] += 1.0
        self.vocab[self.args.mem_tok] += 1.0
        # self.vocab[self.args.rel_tok] += 1.0
        self.vocab[self.args.eos_tok] += 1.0
        # for o in self.objects:
        #     self.vocab[o] += 1.0

        self.stoi = dict(zip(self.vocab.keys(), range(1, len(self.vocab)+1)))
        self.stoi[self.args.pad_tok] = 0
        # add additional tokens
        # self.stoi[self.args.unk_tok] = len(self.stoi)
        # self.stoi[self.args.sos_tok] = len(self.stoi)
        # self.stoi[self.args.eos_tok] = len(self.stoi)
        # print(len(self.stoi))
        # self.itos = {v: k for k, v in self.stoi.items()}

        for j in range(self.max_out_reln):
            self.stoi['o'+str(j)] = len(self.stoi)+1
        # del self.stoi
        self.itos = {v: k for k, v in self.stoi.items()}
        print (len(self.stoi))
        self.n_words = len(self.stoi)

        self.vectors = np.zeros((len(self.stoi), vec_dim))
        for w, w2i in self.stoi.items():
            if w2i < self.stoi[self.args.eos_tok]:
                self.vectors[w2i] = self.word_emb.wv[w]

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
                        if w not in self.objects:
                            self.vocab[w] += 1.0

    def get_conv(self, filename):
        '''
        load all the training data
        :param filename:
        :return:
        '''
        with open(filename, 'r') as inp:
            out_f = json.load(inp)

        return out_f

    def load_all_files(self, dataset):
        '''
        Get list of all files in the directory
        :param dataset:
        :return:
        '''
        dat_files = os.listdir(self.data_path + '/conversations/' + dataset + '_with_entities_correct')
        return dat_files

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
            return self.args.unk_tok

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300)
        if phrase=="":
            return vec
        phrase = phrase.strip()  # Remove _ during similarity calculations
        for w in phrase.split():
            if w not in self.stop and w not in self.punc:
                vec = vec + self.word_emb.wv[w]
        return vec / float(len(phrase.split()))

    # @staticmethod
    def get_kg(self, kg_path):
        '''
        Get the KG dictionary
        :param file_name:
        :return:
        '''
        kg_files = os.listdir(kg_path)
        er_pair_dict = {}
        ent_l = []
        for kgf in kg_files:
            with open(kg_path + kgf, 'r') as f:
                kg = f.readlines()
            for t in kg:
                e, r, o = t.replace('\n', '').split('\t')
                er_pair_dict[e+'#'+r] = o
                er_pair_dict[o+'#'+r] = e
                ent_l.append(e)
                ent_l.append(o)
        return er_pair_dict, ent_l

    @staticmethod
    def cosine_similarity(a, b):
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return sim if not np.isnan(sim) else 0.0

    def calculate_similarity(self, er_elem, query):
        """
        :param er_elem: an entity/relation
        :param query: question from an utterance
        :return: cosine similarity value between query and entity/relation
        """
        er_elem_emb = self.get_avg_word2vec(er_elem)
        query_emb = self.get_avg_word2vec(query)
        score = self.cosine_similarity(er_elem_emb, query_emb)
        return score

    def get_data(self, dataset):
        '''
        Load conversations from files
        :param dataset:
        :return:
        '''
        data_out = []
        found = []
        data_files = self.load_all_files(dataset)
        for dat in data_files:
            # print (dat)
            # print ("dat is here")
            convo = self.get_conv(self.data_path + 'conversations/' + dataset + '_with_entities_correct/' + dat)  # Get Conversations

            conv_hist = []
            for j, c in enumerate(convo):
                convo_dict = {}
                # conv_hist = []
                if conv_hist:
                    conv_hist.append(c['q' + str(j + 1)])
                    convo_dict['q'] = (' '+self.args.eou_tok+' ').join(u for u in conv_hist)
                    conv_hist.append(c['a' + str(j + 1)])
                else:
                    convo_dict['q'] = c['q' + str(j + 1)]
                    conv_hist.append(c['q' + str(j + 1)])
                    conv_hist.append(c['a' + str(j + 1)])
                convo_dict['_a'] = c['a' + str(j + 1)]
                convo_dict['_q'] = c['q' + str(j + 1)]
                convo_dict['a'] = c['a' + str(j + 1)]
                convo_dict['true_ent'] = c['kvr_entlist_qa' + str(j + 1)]
                # conv_hist.append(convo_dict['q']+self.args.eou_tok)
                # conv_hist.append(convo_dict['a']+self.args.eou_tok)
                convo_dict['kgER'] = OrderedDict(c['kgER' + str(j + 1)])
                s = list(c['correct_ERcomb'].keys())  # Get the subject
                if s:
                    s = s[0]
                    r_s = list(c['correct_ERcomb'][s].keys())  # Get all relations
                    o = [c['correct_ERcomb'][s][r] for r in r_s]  # Get all correct objects
                    convo_dict['object'] = o
                    convo_dict['subject'] = s
                    convo_dict['relations'] = r_s
                    # convo_dict['q'] = convo_dict['q'].replace(s, self.args.ent_tok)
                    inp_kg = convo_dict['kgER']
                    # Correct er pair list
                    correct_er_pair = []
                    # Get output relation
                    entity_reln_pair = []
                    correct_ent = []
                    relations = []
                    for ent in inp_kg.keys():
                        for r in inp_kg[ent]:
                            entity_reln_pair.append([ent, r])  # get all entity relation pair
                            relations.append(r)
                    # if 'relation' in data:
                    object_dict = {}
                    for p, er_pair in enumerate(entity_reln_pair):
                        e, r = er_pair
                        try:
                            # object_dict['o'+str(p)] = self.er_dict[e+'#'+r]
                            object_dict['o'+str(p)] = e+'#'+r
                        except KeyError:
                            print (dat)
                        if e == s and r in r_s:
                            correct_er_pair.append(p)
                            correct_ent.append(e+'#'+r)
                    convo_dict['output_kg'] = correct_er_pair
                    convo_dict['er_pair'] = object_dict
                    # Replace answer with object
                    if len(correct_er_pair) > 0:  # check if correct er pair exists in answer
                        sent_gate = np.zeros(len(convo_dict['_a'].split()))
                        answer = []
                        # answer_entity =
                        # for j, c_p in enumerate(correct_er_pair):
                        #     answer = convo_dict['a'].replace(o[j], 'o' + str(c_p))
                        # answer = [convo_dict['a'].replace(o[j], 'o' + str(correct_er_pair[j])) for j in range(len(correct_er_pair))]
                        # sent_gate[b] = 1.0
                        for k, w_a in enumerate(convo_dict['_a'].split()):
                            # print(self.get_w2i(w_a))
                            # y[b, k] = self.get_w2i(w_a)
                            if w_a in o:
                                try:
                                    sent_gate[k] = 1
                                    j = [i for i, obj in enumerate(o) if obj==w_a]
                                    # answer.append(self.args.mem_tok)
                                    answer.append('o' + str(correct_er_pair[j[0]]))
                                except Exception as e:
                                    print (sent_gate, convo_dict['_a'])
                                    print (e)
                            else:
                                answer.append(w_a)
                            # if w_a in self.global_ent:
                            #     correct_ent.append(w_a)
                        convo_dict['a'] = ' '.join(w for w in answer)
                        convo_dict['correct_ent'] = correct_ent
                    try:
                        convo_dict['sent_gate'] = sent_gate
                    except Exception as e:
                        print (e, correct_er_pair, convo_dict['q'])
                    # Get Degree Matrix
                    A, I = gen_adjacency_mat(convo_dict['kgER'])
                    D = torch.from_numpy(get_degree_matrix(A))
                    # get entity relation matrix
                    X = []
                    er_vec = getER_vec(convo_dict['kgER'])
                    for e, ele in enumerate(er_vec):
                        # ele_present = get_fuzzy_match(ele, convo_dict['q'])[0]/100
                        # X.append([self.calculate_similarity(ele, convo_dict['q']), ele_present])
                        # ele_present = 1.0 if ele in convo_dict['q'] else 0.0
                        ele_sim = self.calculate_similarity(ele, convo_dict['_q'])
                        X.append(ele_sim)
                    # Calculate features
                    A_hat = A + I
                    dt = np.matmul(np.linalg.inv(D), A_hat)
                    h = np.matmul(dt, X)
                    convo_dict['X_feat'] = X
                    inp_ft = []
                    all_similarities = []
                    for k, e in enumerate(er_vec):
                        all_similarities.append([e, h[k]])
                    for s in all_similarities:
                        ele, sim = s
                        if ele not in inp_kg:
                            inp_ft.append(sim)
                        else:
                            ent_sim = sim
                    convo_dict['input_graph_feature'] = inp_ft
                    if correct_er_pair:
                        found.append(correct_er_pair)
                        # print (correct_sub, convo_dict['q'], correct_reln, dat)
                data_out.append(convo_dict)
        print (len(found))
        return data_out


if __name__ == '__main__':

    preproc = Preprocessor()
    # file names
    train_preproc_file = preproc.data_path+'preproc_files/train.npy'
    valid_preproc_file = preproc.data_path+'preproc_files/val.npy'
    test_preproc_file = preproc.data_path+'preproc_files/test.npy'
    stoi_f = preproc.data_path+'preproc_files/stoi.npy'
    w2emb_f = preproc.data_path+'preproc_files/wemb.npy'
    # Save all files
    np.save(train_preproc_file, preproc.train_dataset)
    np.save(valid_preproc_file, preproc.val_dataset)
    np.save(test_preproc_file, preproc.test_dataset)
    np.save(stoi_f, preproc.stoi)
    np.save(w2emb_f, preproc.vectors)

