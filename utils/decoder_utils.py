from collections import defaultdict
from utils.args import get_args
from utils.utils_graph import gen_adjacency_mat, get_degree_matrix, getER_vec
import os
import numpy as np
from utils.batcher.incar_batcher_sep_vocab_bert import InCarBatcher
from nltk.corpus import stopwords
import string
import torch
from unidecode import unidecode
args = get_args()


class DecodeSentences:
    def __init__(self, chatdata, data_path='data/incar/', domain='incar'):
        self.chat_data = chatdata
        self.data_path = data_path
        self.word_emb = chatdata.word_emb
        self.domain = domain
        self.stop = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.soccer_conv_map = np.load(os.getcwd()+'/data/convfile2kg_mapping.npy', allow_pickle=True).item()  #  Load conv mapper for soccer dialogues
        # self.er_dict, self.global_ent, self.eo_dict, self.e_r_l = self.chat_data.get_kg(self.data_path + 'KG/',
        # dat=self.domain)

    def _get_sentences(self, sent_indexed):
        out_sents = [self.get_sent(sent_indexed[i]) for i in range(len(sent_indexed))]
        out_sents = [str(sent.split(args.eos_tok)[0]) for sent in out_sents]
        return out_sents

    def calculate_similarity(self, er_elem, query):
        """
        :param er_elem: an entity/relation
        :param query: question from an utterance
        :return: cosine similarity value between query and entity/relation
        """
        # print (er_elem)
        er_elem_emb = self.get_avg_word2vec(er_elem)
        query_emb = self.get_avg_word2vec(query)
        score = self.cosine_similarity(er_elem_emb, query_emb)
        return score

    @staticmethod
    def cosine_similarity(a, b):
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return sim if not np.isnan(sim) else 0.0

    def get_w2v(self, word):
        # get word2vecs
        if '_' not in word:
            return self.word_emb.wv[word]
        else:
            return self.get_avg_word2vec(word.replace('_', ' '))

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300)
        len_phr = 0.0
        if phrase=="":
            return vec
        # phrase = phrase.strip()  # Remove _ during similarity calculations
        for w in phrase.split():
            if w not in self.stop and w not in self.punc:
                vec = vec + self.word_emb.wv[w]
                len_phr += 1.0
        return np.divide(vec, len_phr)

    def get_graph_lap(self, entity, question):
        # orig_g = orig_g.data.numpy()
        i_g = np.ones(len(self.chat_data.trg_stoi))
        # query = ' '.join(self.chat_data.src_itos[w.item()] for w in question)
        # graph_ele = [self.chat_data.trg_stoi[g] for g in self.chat_data.trg_stoi.keys() if '@' in g]
        # remove all elements from vocabulary except at k-hop
        # graph_ele = [self.chat_data.trg_stoi[g] for g in self.chat_data.trg_stoi.keys() if '@' in g or g in self.chat_data.e_r_l.keys()]
        # print (graph_ele)
        # i_g[graph_ele] = 0.0
        input_kg_reln = self.chat_data.e_r_l[entity]
        input_kg = dict()
        input_kg[entity] = list(input_kg_reln)
        A, I = gen_adjacency_mat(input_kg)
        D = get_degree_matrix(A)
        X = []
        # get entity relation matrix
        er_vec = getER_vec(input_kg)
        for e, ele in enumerate(er_vec):
            # ele_present = get_fuzzy_match(ele, convo_dict['q'])[0]/100
            # X.append([self.calculate_similarity(ele, convo_dict['q']), ele_present])
            X.append(self.calculate_similarity(ele, question))
        # Calculate features
        A_hat = A + I
        try:
            dt = np.matmul(np.linalg.inv(D), A_hat)
            h = np.matmul(dt, X)
        except Exception as e:
            # print(e)
            h = np.zeros(len(er_vec))
            # print(dat, A)

        # print(h)  # vector for graph Laplacian

        # kg_dict = {}
        for j, ele in enumerate(er_vec):
            if j == 0:
                try:
                    i_g[self.chat_data.trg_stoi['@entity']] = h[0]
                except KeyError:
                    i_g[self.chat_data.trg_stoi['@entity']] = 1.0
            else:
                # if '@@' + ele in convo_dict['a']:  # check for dual 2-hop relation
                #     kg_dict['@@' + ele] = h[j]
                # else:
                try:
                    ele = ele.replace(' ', '_')
                    i_g[self.chat_data.trg_stoi['@' + ele]] = h[j]
                except KeyError:
                    try:
                        i_g[self.chat_data.trg_stoi['@@' + ele]] = h[j]
                    except Exception:
                        # print (ele, entity)
                        continue

        return torch.from_numpy(i_g)

    def get_sentences(self, sent_indexed, pred_entities, local_kg):
        out_sents = [self.get_sent_obj(sent_indexed[i], pred_entities[i], local_kg[i], dat=self.domain) for i in
                     range(len(sent_indexed))]
        sentences = [str(sent.split(args.eos_tok)[0]) for sent, pred_ent, pred_reln, p_s_orig, kg_l in out_sents]
        sentences_orig = [str(p_s_orig.split(args.eos_tok)[0]) for sent, pred_ent, pred_reln, p_s_orig, kg_l in out_sents]
        pred_entities = [pred_ent for sent, pred_ent, pred_reln, p_o, kg_l in out_sents]
        pred_relations = [pred_reln for sent, pred_ent, pred_reln, p_o, kg_l in out_sents]
        kg_entities = [kg_l for sent, pred_ent, pred_reln, p_o, kg_l in out_sents]

        return sentences, pred_entities, pred_relations, sentences_orig, kg_entities

    def get_sent(self, sentence):
        # Get sentences
        out_sent = []
        for w in sentence:
            word = w.item()
            if word > self.chat_data.trg_stoi[args.eos_tok]:
                out_sent.append(self.chat_data.get_i2w(word)+'<ent>')
            else:
                out_sent.append(self.chat_data.get_i2w(word))
        return ' '.join(out_sent)

    def get_sent_obj(self, sentence, entity, local_kg, dat='soccer'):
        # Local KG load
        # stoi = self.chat_data.trg_itos
        # itos = self.chat_data.trg_stoi
        # print(sentence)
        sentence = sentence[0]
        # print(sentence, entity)
        if dat=='soccer':  # for soccer check in both
            kg = self.soccer_conv_map[local_kg]
            if kg: # check if kg exists or a generic conversation
                try:
                    with open(self.data_path+'KG/clubs/'+kg+'_kg.txt', 'r') as f:
                        kg = f.readlines()
                except FileNotFoundError:
                    with open(self.data_path+'KG/country/'+kg+'_kg.txt', 'r') as f:
                        kg = f.readlines()
            else:
                kg = []
        else:
            with open(self.data_path+'KG/'+local_kg+'_kg.txt', 'r') as f:
                kg = f.readlines()
        _1hop_r = defaultdict()
        kg_ent = []
        for t in kg:
            e, r, o = unidecode(t).lower().replace('\n', '').split('\t')
            kg_ent.append(e)
            _1hop_r[e+'#'+r] = o
            if 'weather' not in local_kg:
                _1hop_r[o+'#'+r] = e

        # Get sentences
        out_sent = []
        predicted_obj = set()
        relations = []
        predicted_ir_sent = [self.chat_data.trg_itos[w.item()] for w in sentence]  # Get intermediate representation
        predicted_ent = self.chat_data.itoe[entity.item()]
        # predicted_ent = entity[0]
        if dat == 'incar':
            predicted_obj.add(predicted_ent)
        # predicted_ent = entity
        if dat=='soccer':  # Handle separately for soccer and incar
            for w in predicted_ir_sent:
                if '@' in w:
                    if '@entity' in w:
                        out_sent.append(predicted_ent)
                    else:
                        print('predicting token from kg', w)
                        if args.no_ent_tok not in predicted_ent:
                            predicted_reln = w.replace('@', '').replace('_', ' ')
                            try:
                                relations.append(predicted_reln)
                                print(predicted_ent, predicted_reln)
                                obj = _1hop_r[predicted_ent + '#' + predicted_reln]
                                print(obj)
                                out_sent.append(obj)
                                predicted_obj.add(obj)
                            except KeyError:
                                out_sent.append(w)
                        else:
                            out_sent.append(w)
                else:
                    out_sent.append(w)
        else:
            for w in predicted_ir_sent:
                if '@@' in w:
                    try:
                        connected_obj = _1hop_r[predicted_ent + '#poi']
                        try:
                            connected_obj_2hp = _1hop_r[connected_obj + '#' + w.replace('@', '')]
                        except KeyError:
                            connected_obj_2hp = _1hop_r[connected_obj + '#' + w.replace('@@', '')]
                    except KeyError:
                        try:
                            # connected_obj = _1hop_r[predicted_ent + '#poi']
                            connected_obj_2hp = _1hop_r[predicted_ent + '#' + w.replace('@@', '')]
                        except KeyError:
                            connected_obj = w
                            connected_obj_2hp = w
                        # connected_obj = w
                        # connected_obj_2hp = w
                    out_sent.append(connected_obj_2hp)
                    predicted_obj.add(connected_obj_2hp)
                elif '@' in w:
                    if '@entity' in w:
                        out_sent.append(predicted_ent)
                    else:
                        if args.no_ent_tok not in predicted_ent:
                            predicted_reln = w.replace('@', '')
                            try:
                                relations.append(predicted_reln)
                                obj = _1hop_r[predicted_ent + '#' + predicted_reln]
                                out_sent.append(obj)
                                predicted_obj.add(obj)
                            except KeyError:
                                out_sent.append(w)
                        else:
                            out_sent.append(w)
                else:
                    out_sent.append(w)

        return ' '.join(out_sent[1:]), list(predicted_obj), list(relations), ' '.join(predicted_ir_sent), kg_ent  # Remove SOS token


if __name__ == '__main__':
    chat_data = InCarBatcher()
    sentence_decoder = DecodeSentences(chat_data)
    er_vec = ['titanic', 'year', 'directed by']
    question = 'who is the director of titanic ?'
    X = []
    A, I = gen_adjacency_mat({'titanic': ['year', 'directed by']})
    D = get_degree_matrix(A)
    # er_vec = getER_vec(input_kg)
    for e, ele in enumerate(er_vec):
        # ele_present = get_fuzzy_match(ele, convo_dict['q'])[0]/100
        # X.append([self.calculate_similarity(ele, convo_dict['q']), ele_present])
        X.append(sentence_decoder.calculate_similarity(ele, question))
    # Calculate features
    A_hat = A + I
    # try:
    dt = np.matmul(np.linalg.inv(D), A_hat)
    h = np.matmul(dt, X)
    # sim = sentence_decoder.calculate_similarity('directed by', 'who is the director of titanic ?')
    # print (sim)
    print (X)
    print (h)
    print (A)