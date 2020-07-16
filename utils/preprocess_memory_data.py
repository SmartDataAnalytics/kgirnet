import numpy as np
from unidecode import unidecode
import torch
import os
import json
import csv
import sys
import re
import spacy
from tqdm import tqdm
import logging
from spacy.tokenizer import Tokenizer
from generate_matrix import MemoryGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])
logger = logging.getLogger()

#spacy tokenizers
nlp = spacy.load('en')
pos = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp.vocab)

class Preprocessing():
    def __init__(self, dataset, emb_path):

        logger.info("Loading Vocab, Knowledge Graphs, conversation to kg file mapping ...")
        self.dataset_name = dataset
        self.conv2kg = self.conv2kg_mapping()
        self.maxkbsize = self.get_max_kb(self.dataset_name, saveKG=True)
        self.mapping = json.load(open("data/"+self.dataset_name+"/ERmapping.json"))
        self.kgs = np.load("preproc_files/"+self.dataset_name+"/kgs.npy", allow_pickle=True).item()
        self.w2i = self.load_vocab(self.dataset_name)

        self.memgen = MemoryGenerator(self.dataset_name, self.conv2kg,self.kgs, fasttext_emb_path=emb_path)
        print("MAX ENT,REL: ",self.memgen.maxEntity, self.memgen.maxRel)
        self.triandata, self.testdata, self.valdata =  self.build_memorydata(self.dataset_name, save2file=True)


    def build_memorydata(self, dataset, save2file=False):
        datadir = "data/"+dataset+"/conversations/"
        datatypes = ["train","val","test"]
        #datatypes = ["val"]
        memory = {}
        corrected_files = {}

        if dataset=="soccer":
            for dtype in datatypes:
                corrected_files[dtype] = self.corrected_info(dtype)

        for dtype in datatypes:
            total_found = 0
            if dataset=="incar":
                filedir = datadir+dtype+"_without_buboqa/"
            else:
                filedir = datadir + dtype + "_with_entities_r/"
            memory[dtype] = []
            if dataset=="soccer":
                for convfile in os.listdir(filedir):
                    filepath = filedir+convfile
                    f = convfile.replace(".json","")
                    kgfile = self.conv2kg[f]
                    if kgfile + "_kg" in self.kgs:
                        if convfile in corrected_files[dtype][0][0]:
                            memory[dtype].append(self.calc_matrices(filedir+convfile, dataset, True, corr_info= corrected_files[dtype]))
                        else:
                            memory[dtype].append(self.calc_matrices(filedir+convfile, dataset,False,corr_info=corrected_files[dtype]))

            else:
                print("Incar data "+20*'-'+dtype.upper())
                for idx, convfile in tqdm(enumerate(os.listdir(filedir))):
                    memory[dtype].append(self.calc_matrices(filedir+convfile, dataset))
            # print("Dtype: ", dtype, total_found)
            # exit()

        if save2file:
            for dtype in datatypes:
                np.save("preproc_files/"+dataset+"/"+dtype+".npy", memory[dtype])

        return memory["train"], memory["test"], memory["val"]

    def find_corr_rel(self, correct_inf, f,q):
        for i,inf in enumerate(correct_inf[0][1]):
            if inf.strip()==self.clean_str(q):
                return True, correct_inf[0][2][i], correct_inf[0][3][i]
        return False, "",""

    def get_entity_from_kg(self,ent,rel):
        for kg in self.kgs.values():
            for i in range(len(kg[0])):
                if kg[1][i].lower()==rel and kg[2][i].lower()==ent:
                    return True,kg[0][i].lower()
        return False, ""

    def calc_matrices(self, filepath, dataset, found,corr_info=[]):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        #logger.info(filepath)
        Ques = []
        Ans = []
        Ans_entities = []
        H = []
        H_exp = []
        Adj_matrix = []
        D_inv = []
        Weight_vec  = []
        out_correct = []
        nodes_with_ER = []
        existing_conn = []

        f = filepath[filepath.rfind("/")+1:filepath.find(".")]
        if dataset=="incar":
            kg = self.kgs[f+"_kg"]
        else:
            kgfile = self.conv2kg[f]
            kg = [val for ke,val in self.kgs.items()]

        totalfound=0
        for i,utt in enumerate(data):
            entities = []
            relations = []
            er_dict = {}

            if dataset=="soccer":
                global_ents = set(self.mapping.keys())

                Ques.append(utt["q" + str(i + 1)])
                Ans.append(utt["a" + str(i + 1)])
                if found:
                    status,rr,aa = self.find_corr_rel(corr_info,f,utt["q"+str(i+1)])
                    if status:
                        totalfound+=1
                        ent_status, found_ent = self.get_entity_from_kg(aa.lower(),rr.lower())
                        if ent_status:
                            entities.append(found_ent)
                            relations  = self.mapping[found_ent]
                            er_dict[found_ent] = relations
                            print(entities,relations)
                        else:
                            #triple Not found in kg
                            print("DID NOT FOUND FOR: ",kgfile,aa,rr)
                    else:
                        #utterence Not found in corrected file
                        pass
                else:
                    #Not found in corrected file too
                    pass

            else:
                for ent,rels in utt["kgER_e"+str(i+1)].items():
                    ent = '_'.join(ent.split())
                    rels = ['_'.join(r.split()) for r in rels]
                    if ent not in entities:
                        er_dict[ent] = rels
                        entities.append(ent)
                        for r in rels:
                            if r not in relations:
                                relations.append(r)
                        if len(rels)==1 and rels[0]=="poi_type":
                            for j in range(len(kg[0])):
                                if kg[1][j] == "poi_type" and kg[2][j] == ent:
                                    if kg[0][j] not in entities:
                                        entities.append(kg[0][j])
                                        er_dict[kg[0][j]] = self.mapping[kg[0][j]]
                                    for rr in self.mapping[kg[0][j]]:
                                        if rr not in relations:
                                            relations.append(rr)

            #Now searching for entities in ans
            ans_entities = []
            global_ents = set(self.mapping.keys())
            for ent in global_ents:
                if ent in utt["a"]+str(i+1):
                    ans_entities.append(ent)
            #ans_entities = [ent for ent in utt["a"+str(i+1)].split() if ent in global_ents]

            #logger.log("Q: "+ utt["q"+str(i+1)])
            Ques.append(utt["q"+str(i+1)])
            #logger.log("A: "+utt["a"+str(i+1)])
            Ans.append(utt["a"+str(i+1)])
            er_vec = self.memgen.getER_vec(entities,relations)

            ansvec = {}
            print("Entities: ", entities)
            print("Relations: ", relations)
            for p,a in enumerate(entities):
                for q,b in enumerate(relations):
                        ansvec[a+"$"+str(p)+"#"+b+"$"+str(q)] = 0

            A = self.memgen.get_adjacency_matrix(er_vec, er_dict)
            if A is None:
                return None
            print("A")
            print("ER_DICT: ",er_dict)
            Adj_matrix.append(A)
            A_hat = A + np.identity(A.shape[0])
            D = self.memgen.get_degree_matrix(A)
            X = self.memgen.calc_weight_vector(er_vec, utt["q"+str(i+1)])
            print("A_hat:")
            print(A_hat)
            print("ANS VEC:", ansvec)
            print("X:")
            print(X)
            Weight_vec.append(X)
            tm = np.linalg.inv(D)
            D_inv.append(tm)
            dt = np.matmul(tm, A_hat)

            #logger.info("H = D**-1 x A x X --> "+str( h))
            outputvec = np.zeros(self.memgen.maxRel * self.memgen.maxEntity)
            if dataset=="incar":
                correct_rel = set()
                for k,er in enumerate(er_vec):
                    if er in er_dict: #"entity"
                        er_rels = er_dict[er]
                        if len(er_rels)==1: #Then it is probably in the right side of a triple (Object)
                            for j in range(len(kg[0])):
                                if kg[1][j] == er_rels[0] and kg[2][j] == er:
                                    if kg[0][j] in ans_entities:    #if entites exists in ans_enties then found answerable triple
                                        correct_rel.add(er_rels[0])
                                        for m,n in ansvec.items():
                                            splt = m.split("#")
                                            if er in splt[0] and er_rels[0] in splt[1]:
                                                ansvec[m] = 1
                        else:
                            for arel in er_rels:
                                for j in range(len(kg[0])):
                                    if kg[1][j] == arel and kg[0][j] == er:
                                        if kg[2][j] in ans_entities:  # if entities exists in ans_entities then found answerable triple
                                            correct_rel.add(arel)
                                            for m, n in ansvec.items():
                                                splt = m.split("#")
                                                if er in splt[0] and arel in splt[1]:
                                                    ansvec[m] = 1
            else:
                for comb in ansvec.keys():
                    splt = comb.split("#")
                    if entities[0] in splt[0] and rr in splt[1]:
                        ansvec[comb] = 1
                    if rr in splt[0] and entities[0] in splt[1]:
                        ansvec[comb] = 1

            nodes_with_ER.append(list(ansvec.keys()))
            #CHECKING EXISTING CONNECTION
            existing_connection_msk = []
            for k in list(ansvec.keys()):
                tmp_split = k.split("#")
                e_ = tmp_split[0].split("$")[0]
                r_ = tmp_split[1].split("$")[0]
                if r_ in self.mapping[e_]:
                    existing_connection_msk.append(1)
                else:
                    existing_connection_msk.append(0)

            # Modified calculation of H
            h = np.matmul(dt,X)
            #h = np.multiply(h,existing_connection_msk)         # pointwise multiplication
            expanded_h = self.expand_H(h,len(entities), len(relations))
            expanded_h = np.multiply(expanded_h,np.array(existing_connection_msk))
            H.append(h)
            print("H: ", h)
            H_exp.append(expanded_h)
            print("avg H: ", expanded_h)
            existing_conn.append(np.array(existing_connection_msk))
            outputvec = np.array([v for k,v in ansvec.items()])

            out_correct.append(outputvec)
            Ans_entities.append(ans_entities)
            print("Ans Entities: ", ans_entities)
            print("OUTPUT_vector: ",len(outputvec),outputvec)

        if totalfound!=0:
            # print(out_correct)
            # print(er_vec)
            pass
        fname = filepath[filepath.rfind("/")+1:filepath.find(".")]
        kgname = self.conv2kg[fname]
        kg = self.kgs[kgname+"_kg"]
        return Ques,Ans, Ans_entities,H_exp,nodes_with_ER,out_correct,kg,kgname+"_kg", Adj_matrix, D_inv, Weight_vec, H, existing_conn


    def expand_H(self,H,n_ent,n_rel):
        expanded_H = []
        for i in range(n_ent):
            for j in range(n_ent,n_ent+n_rel):
                    expanded_H.append(np.average([H[i],H[j]]))
        return np.array(expanded_H)


    def corrected_info(self, datatype, dataset="soccer"):
        info = []
        files, q, correct_rel, ans = [], [], [], []
        with open("data/"+dataset+"/correct_data/"+datatype+"_kgqa.csv",encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                files.append(row[0]+".json")
                q.append(self.clean_str(row[1]))
                correct_rel.append(self.clean_str(row[3]))
                ans.append(self.clean_str(row[4]))
        info.append([files, q, correct_rel, ans])
        return info


    def get_max_kb(self, dataset, saveKG=False):
        allkgs  = {}
        kgtypes = ["clubs","country"] if dataset=="soccer" else ["incar"]
        for kgtype in kgtypes:
            kgdir = "data/"+dataset+"/KG/"+kgtype+"/"
            for kg in os.listdir(kgdir):
                allkgs[kg.replace('.txt','')] = self.readKG(kgdir+kg)
        max_len = np.max([(len(a)) for a,b,c in allkgs.values()])
        if saveKG:
            np.save("preproc_files/"+dataset+"/kgs.npy",allkgs)
        return max_len


    def readKG(self,filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            kg_info = f.readlines()
        kg_info = [unidecode(l) for l in kg_info]
        kg_sub = [info.replace('\n', '').split('\t')[0].strip().lower() for info in kg_info]
        kg_reln = [info.replace('\n', '').split('\t')[1].strip().lower() for info in kg_info]
        kg_obj = [info.replace('\n', '').split('\t')[-1].strip().lower() for info in kg_info]
        return kg_sub, kg_reln, kg_obj


    def conv2kg_mapping(self):
        return np.load("data/convfile2kg_mapping.npy", allow_pickle=True).item()


    def load_vocab(self, dataset):
        return np.load("data/soccer/preproc_files/stoi.npy", allow_pickle=True).item()

    def getw2id(self, word):
        return self.w2i['unk'] if word not in self.w2i else self.w2i[word]

    def getsent2i(self,sent):
        tokens = tokenizer(sent.strip())
        out = [self.getw2id(t.text) for t in tokens]
        return out

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
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()


fasttext_emb = "data/wiki.simple.bin"
pp = Preprocessing('soccer', emb_path=fasttext_emb)
logger.info("Max KG size "+str(pp.maxkbsize))
