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
import pandas as pd
from utils.preprocessor import clean_str, get_fuzzy_match, fuzz
from unidecode import unidecode
from collections import OrderedDict
from utils.args import get_args


class Preprocessor:
    '''
    Preprocessor class
    '''
    def __init__(self, data_path='data/soccer/', vec_dim=300,
                 # fasttext_model='/home/debanjan/acl_submissions/soccerbot_acl/vocab/wiki.simple.bin'):
                 fasttext_model='/data/dchaudhu/soccerbot_acl/vocab/wiki.en.bin'):
        self.data_path = data_path
        self.max_similarity = 85
        self.vec_dim = vec_dim
        self.args = get_args()
        cap_path = datapath(fasttext_model)
        self.word_emb = load_facebook_model(cap_path)
        # print (self.max_er_vec)
        self.stop = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.ent_d, self.ent_list = self.get_kg(self.data_path+'KG/')
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
        self.get_vocab(self.train_dataset)
        self.get_vocab(self.test_dataset)
        self.get_vocab(self.val_dataset)
        self.vocab[self.args.unk_tok] += 1.0
        self.vocab[self.args.sos_tok] += 1.0
        self.vocab[self.args.eou_tok] += 1.0
        self.vocab[self.args.mem_tok] += 1.0
        self.vocab[self.args.eos_tok] += 1.0
        for o in self.objects:
            self.vocab[o] += 1.0
        # self.stoi[self.args.pad_tok] = 0
        self.stoi = dict(zip(self.vocab.keys(), range(1, len(self.vocab)+1)))
        self.stoi[self.args.pad_tok] = 0
        # add additional tokens
        # self.stoi[self.args.unk_tok] = len(self.stoi)
        # self.stoi[self.args.sos_tok] = len(self.stoi)
        # self.stoi[self.args.eos_tok] = len(self.stoi)
        # print(len(self.stoi))
        # self.itos = {v: k for k, v in self.stoi.items()}

        # for j in range(self.max_out_reln):
        #     self.stoi['o'+str(j)] = len(self.stoi)+1
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

    # @staticmethod
    def get_kg(self, kg_path):
        '''
        Get the KG dictionary
        :param file_name:
        :return:
        '''
        kg_files_club = os.listdir(kg_path+'clubs/')
        kg_files_nation = os.listdir(kg_path+'country/')
        er_pair_dict = {}
        ent_l = []
        for kgf in kg_files_club:
            with open(kg_path+'clubs/'+kgf, 'r') as f:
                kg = f.readlines()
            for t in kg:
                e, r, o = t.lower().replace('\n', '').split('\t')
                e, r, o = clean_str(e), clean_str(r), clean_str(o)
                er_pair_dict[e.strip()+'#'+r.strip()] = o.replace(' ', '_')
                er_pair_dict[o.strip()+'#'+r.strip()] = e.replace(' ', '_')
                ent_l.append(e)
                ent_l.append(o)
        for kgf in kg_files_nation:
            with open(kg_path+'country/'+kgf, 'r') as f:
                kg = f.readlines()
            for t in kg:
                e, r, o = t.lower().replace('\n', '').split('\t')
                e, r, o = clean_str(e), clean_str(r), clean_str(o)
                er_pair_dict[e.strip() + '#' + r.strip()] = o.replace(' ', '_')
                er_pair_dict[o.strip() + '#' + r.strip()] = e.replace(' ', '_')
                ent_l.append(e)
                ent_l.append(o)

        return er_pair_dict, ent_l

    def load_all_files(self, dataset):
        '''
        Get list of all files in the directory
        :param dataset:
        :return:
        '''
        dat_files = os.listdir(self.data_path + '/conversations/' + dataset + '_with_entities_er')
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

    def get_kgqa(self, dataset):
        # Get the input KG
        kgqa_dict = defaultdict(list)
        kgqa_dat = np.array(pd.read_csv(self.data_path+'correct_data/'+dataset+'_kgqa.csv', header=None))
        for v in kgqa_dat:
            c, q, a, r, s = v
            kgqa_dict[c].append([q, a, r, s])
        return kgqa_dict

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300)
        if phrase=="":
            return vec
        phrase = unidecode(phrase).strip()
        for w in phrase.split():
            if w not in self.stop and w not in self.punc:
                vec = vec + self.word_emb.wv[w]
        return vec / float(len(phrase.split()))

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
        kgqa_answers = self.get_kgqa(dataset)
        for dat in data_files:
            # print (dat)
            # print ("dat is here")
            convo = self.get_conv(
                self.data_path + 'conversations/' + dataset + '_with_entities_er/' + dat)  # Get Conversations
            conv_hist = []
            for j, c in enumerate(convo):
                convo_dict = {}
                r, o = self._check_kgqa_ans(kgqa_answers, dat.replace('.json', ''), c['q' + str(j + 1)])
                if conv_hist:
                    conv_hist.append(c['q' + str(j + 1)])
                    convo_dict['q'] = (' '+self.args.eou_tok+' ').join(u for u in conv_hist)
                    conv_hist.append(c['a' + str(j + 1)])
                else:
                    convo_dict['q'] = c['q' + str(j + 1)]
                    conv_hist.append(c['q' + str(j + 1)])
                    conv_hist.append(c['a' + str(j + 1)])
                convo_dict['file_name'] = dat
                convo_dict['a'] = clean_str(c['a' + str(j + 1)])
                convo_dict['_a'] = clean_str(c['a' + str(j + 1)])
                convo_dict['kgER'] = OrderedDict(c['kgER' + str(j + 1)])
                correct_ent = []
                correct_sub = ''
                if r:
                    # convo_dict['relation'] = r
                    convo_dict['object'] = o
                    # if dataset == 'train':
                    probable_reln = []
                    #inp_kg = c['kgER_e' + str(j + 1)]
                    inp_kg = convo_dict['kgER']
                    # if dat == '3S8A4GJRD4FZ3YXJ1VN53CDZINRV6E.json':
                    # for e_r in c['kgER_e'+str(j+1)]:
                    if len(inp_kg.keys()) > 1:  # More than 1 entity in KG
                        for k, v in inp_kg.items():
                            for r_in_v, kg_r in enumerate(v):
                                if kg_r.lower() == r:
                                    probable_reln.append([k, kg_r.lower()])
                        if len(probable_reln) > 1:
                            best_sub = np.argmax([get_fuzzy_match(opt[0], convo_dict['q'])[0] for opt in probable_reln])
                            correct_sub, correct_reln = probable_reln[best_sub]
                    else:
                        for k, v in inp_kg.items():
                            for r_in_v, kg_r in enumerate(v):
                                if kg_r.lower() == r:
                                    correct_sub, correct_reln = k, kg_r.lower()
                    if not correct_sub:
                        try:
                            correct_sub, correct_reln = probable_reln[0]
                        except Exception as e:
                            # print (dat, '\t', convo_dict['q'])
                            print(e)
                    convo_dict['subject'] = correct_sub
                    convo_dict['relation'] = r
                    # best_sub_match = get_fuzzy_match(correct_sub, convo_dict['a'])[1]
                    # convo_dict['q'] = convo_dict['q'].replace(best_sub_match, self.args.ent_tok)
                    best_match = get_fuzzy_match(o, convo_dict['a'])[1]  # get_fuzzy_match return similarity and object

                    # Get output relation
                    entity_reln_pair = []
                    object_dict = {}
                    relations = []
                    for ent in inp_kg.keys():
                        for r in inp_kg[ent]:
                            entity_reln_pair.append([clean_str(ent), clean_str(r)])  # get all entity relation pair
                            relations.append(r)
                    # if 'relation' in data:
                    for p, er_pair in enumerate(entity_reln_pair):
                        e, r = er_pair
                        try:
                            object_dict['o' + str(p)] = self.ent_d[e+'#'+r]
                        except KeyError:
                            try:
                                object_dict['o' + str(p)] = self.ent_d[e + ' #' + r]
                            except KeyError:
                                object_dict['o' + str(p)] = 'o' + str(p)
                        if e == correct_sub and r == correct_reln:
                            correct_er_pair = p
                            correct_ent.append(self.ent_d[e+'#'+r])
                    convo_dict['output_kg'] = correct_er_pair
                    convo_dict['er_pair'] = object_dict
                    # Replace answer with object
                    if 'correct_er_pair' in vars():
                        sent_gate = np.zeros(len(convo_dict['a'].split()))
                        answer = convo_dict['a'].replace(best_match, self.args.mem_tok)
                        # sent_gate[b] = 1.0
                        for k, w_a in enumerate(answer.split()):
                            # print(self.get_w2i(w_a))
                            # y[b, k] = self.get_w2i(w_a)
                            if w_a == ('o' + str(correct_er_pair)):
                                try:
                                    sent_gate[k] = 1
                                except Exception:
                                    print (sent_gate, convo_dict['a'])
                                    print (sent_gate)
                        convo_dict['a'] = answer
                        convo_dict['_a'] = convo_dict['_a'].replace(best_match, best_match.replace(' ', '_'))
                        # convo_dict['a'] = ' '.join(w for w in answer)
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
                        X.append(self.calculate_similarity(ele, convo_dict['q']))
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
                            inp_ft.append(np.exp(sim))
                        else:
                            ent_sim = sim
                    convo_dict['input_graph_feature'] = inp_ft
                    if correct_sub:
                        found.append(correct_sub)
                        # print (correct_sub, convo_dict['q'], correct_reln, dat)
                data_out.append(convo_dict)
        print (len(found))
        return data_out

    def _check_kgqa_ans(self, kgqa_dict, qid, q):
        '''
        Get relation and subjects for kg based questions
        :param kgqa_dict:
        :param qid:
        :param q:
        :return:
        '''
        correct_reln = ''
        correct_sub = ''
        if qid in kgqa_dict.keys():
            for v in kgqa_dict[qid]:
                q = clean_str(q)
                q_kg = v[0]
                # print (fuzz.ratio(q, q_kg))
                if fuzz.ratio(q, q_kg) > self.max_similarity:
                    # print (q)
                    correct_reln = v[2]
                    correct_sub = v[3]
                    break
                else:
                    continue
        return correct_reln, correct_sub
        # print(len(found))


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

