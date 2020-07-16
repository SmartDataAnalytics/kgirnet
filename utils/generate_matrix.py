import os
import re
from unidecode import unidecode
import numpy as np
import json
import sys
import logging
from numpy.linalg import norm
from gensim.test.utils import datapath
from  gensim.models.fasttext import load_facebook_model
from spacy.lang.en.stop_words import STOP_WORDS

STOP_WORDS.add('de_l_la_le_di')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger()


class MemoryGenerator():
    def __init__(self, dataset, conv2kg, kgs, fasttext_emb_path):
        logger.info("Initializing Memory Generator ....")
        self.conv2kg = conv2kg
        self.kgs = kgs
        self.mapping = json.load(open("data/"+dataset+"/ERmapping.json"))
        self.maxEntity, self.maxRel = self.read_dataset(dataset)
        logger.info("MaxENT: "+str(self.maxEntity)+" maxREL: "+str(self.maxRel))
        self.matrix_dim = self.maxEntity + self.maxRel
        self.word_emb = load_facebook_model(datapath(os.getcwd()+"/"+fasttext_emb_path))
        logger.info("READY: Memory Generator")

    def get_degree_matrix(self, adjacency_matrix):
        """
        :param adjacency_matrix:
        :return: return degree matrix -> example [[1. 0. 0. 0.],
                                                  [0. 2. 0. 0.],
                                                  [0. 0. 0. 0.],
                                                  [0. 0. 0. 3.]]   diagonal numbers represents total number of connections
        """
        return np.array(np.diag(np.array(np.sum(adjacency_matrix, axis=0))))

    def getER_vec(self,entities, relations):
        """
        :param entities: list of entities
        :param relations: list of relations
        :return: generate ER list -> ["brazil","coach","caps",age"]
        """
        er_vec = [e for e in entities]+ [r for r in relations]
        return np.array(er_vec)

    def get_adjacency_matrix(self, er_vector, er_dict):
        """
        :param er_vector: entity/relation vector adjusted to max dimension -> ["brazil","naymar","","","","coach","caps",age",....]
                          NB: can be generated using [get_ER_vecotor() function, given a list of entities and a list of relations]
        :param er_dict: example  {
                                   "Neymar":["age","caps","height",....],
                                   "brazil":["coach","ground",......]
                                    }
        :return: adjacency matrix -> demo example  [[0. 0. 0. 0. 0. 0. 0. 1.]
                                                    [0. 0. 0. 0. 0. 0. 1. 0.]
                                                    [0. 0. 0. 0. 0. 0. 0. 0.]
                                                    [0. 0. 0. 0. 0. 0. 0. 0.]
                                                    [0. 0. 0. 0. 0. 0. 0. 0.]
                                                    [0. 0. 0. 0. 0. 0. 0. 0.]
                                                    [0. 1. 0. 0. 0. 0. 0. 0.]
                                                    [1. 0. 0. 0. 0. 0. 0. 0.]]
        """

        dimension = len(list(er_dict.keys()))+np.sum([len(v) for _,v in er_dict.items()])
        if dimension>0:
            adjacenty_matrix = np.zeros((dimension, dimension))
        else:
            return None
        
        #Forget about param er_vector
        er_vector = []
        for k,v in er_dict.items():
            er_vector.append(k)
            for r in v:
                er_vector.append(r)


        for i in range(dimension):
            if er_vector[i] in er_dict:  # if element of er_vector[i] is an entity
                for j in range(dimension):
                    if er_vector[j] in er_dict[er_vector[i]]:
                        adjacenty_matrix[i][j] = 1.0
            else:                              # if element of er_vector[j] is a relation
                for j in range(dimension):
                    for k in range(dimension):
                        if er_vector[k] in er_dict and er_vector[j] in er_dict[er_vector[k]]:
                            adjacenty_matrix[j][k] = 1.0
        return adjacenty_matrix

    def clean_str(self, string):
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
        return string.strip().lower()

    def get_avg_word2vec(self, phrase):
        """
        get Average word vectors for a given phrases
        """
        vec = np.zeros(300).astype(np.float32)
        if phrase=="":
            return vec
        phrase = unidecode(phrase).strip()
        for w in phrase.split():
            vec = vec + np.array(self.word_emb.wv[w]).reshape(300).astype(np.float32)
        return vec / float(len(phrase.split()))

    def cosine_similarity(self, a, b):
        sim = np.dot(a, b) / (norm(a) * norm(b))
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

    def calc_weight_vector(self,er_vector, query):
        weight_vector = [self.calculate_similarity(er_elem, query) if er_elem != "" else 0.0 for er_elem in er_vector]
        return weight_vector

    def read_dataset(self, dataset):
        data_types = ["train","test","val"]
        max_n_entity = 0
        max_n_rel = 0
        for data_type in data_types:
            datadir = "data/"+dataset+"/conversations/"+data_type+"_with_entities_q/"
            allfiles = os.listdir(datadir)
            for f in allfiles:
                data = json.load(open(datadir+"/"+f,"r", encoding="utf-8"))
                relations = []
                ent_set = set()
                if dataset=="soccer":
                    for data_dict in data:
                        for k,v in data_dict.items():
                            if "kgER" in k:
                                for e, rel_list in v.items():
                                    ent_set.add(e)
                                    relations = relations + rel_list
                else:

                    kg = self.kgs[f.replace(".json", "") + "_kg"]
                    for i, utt in enumerate(data):
                        for ent,rels in utt["kgER"+str(i+1)].items():
                            ent = '_'.join(ent.split())
                            rels = ['_'.join(r.split()) for r in rels]
                            if ent not in ent_set:
                                ent_set.add(ent)
                                if len(rels)==1 and rels[0]=="poi_type":
                                    #get entity for poi_type and add that as an entity and their relations
                                    for i in range(len(kg[0])):
                                        if kg[1][i]=="poi_type" and kg[2][i]==ent:
                                            if kg[0][i] not in ent_set:
                                                ent_set.add(kg[0][i])
                                            for rr in self.mapping[kg[0][i]]:
                                                if rr not in relations:
                                                    relations.append(rr)
                                else:
                                    for r in rels:
                                        if r not in relations:
                                            relations.append(r)

                relations = set(relations)
                max_n_entity = max(max_n_entity,len(ent_set))
                max_n_rel = max(max_n_rel, len(relations))
        return (max_n_entity, max_n_rel)
