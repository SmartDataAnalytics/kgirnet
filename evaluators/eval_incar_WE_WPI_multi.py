from numpy import dot
from numpy.linalg import norm
from collections import defaultdict
from tqdm import tqdm
from unidecode import unidecode
import multiprocessing
import numpy as np
import time
import pulp
from gensim.test.utils import datapath
from  gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors

np.seterr(over='raise')
#print("Loading Word Vectors ..... ")


class Evaluate_we_wpi():
    def __init__(self, embedding_path, vec_dim, references):
        self.word_emb = np.load(embedding_path,allow_pickle=True).item()
        self.vec_dim = vec_dim
        self.references = references
        self.word_freq_ref = self.get_word_frequencies(self.references)
        self.doc_freq_ref = self.get_doc_frequencies(self.references, self.word_freq_ref)

        self.weights_ref = [[self.get_weights(w, self.word_freq_ref, self.doc_freq_ref, len(self.references)) for w in sent.split()]
                       for sent in self.references]
        self.weights_ref = [self.calc_avg_weight(weight) for weight in self.weights_ref]

    def update_reference(self, references):
        self.references = references
        self.word_freq_ref = self.get_word_frequencies(self.references)
        self.doc_freq_ref = self.get_doc_frequencies(self.references, self.word_freq_ref)
        self.weights_ref = [[self.get_weights(w, self.word_freq_ref, self.doc_freq_ref, len(self.references)) for w in sent.split()]
                       for sent in self.references]
        self.weights_ref = [self.calc_avg_weight(weight) for weight in self.weights_ref]

    def get_emb(self, word):
        if word in self.word_emb:
            return np.array(self.word_emb[word]).reshape(200).astype(np.float32)
        else:
            return np.array(np.random.uniform(-0.25,0.25,200)).reshape(200).astype(np.float32)

    def pos_inf(self, pos_ti, pos_ri, m, n):
        pos_i = abs(pos_ti/float(m) - pos_ri/float(n))
        return pos_i

    def cosine_similarity(self, a, b):
        sim = np.dot(a, b) / (norm(a) * norm(b))
        return sim if not np.isnan(sim) else 0.0

    def align_score(self, pos_pred, pos_gold, sent_pred, sent_gold):
        n = len(sent_gold)
        m = len(sent_pred)
        return np.multiply(self.cosine_similarity(self.get_emb(sent_pred[pos_pred-1]), self.get_emb(sent_gold[pos_gold-1]))
               ,(1.0 - self.pos_inf(pos_pred, pos_gold, m, n)))

    @staticmethod
    def calc_avg_weight(weight):
        total = np.sum(weight)
        wt = [np.divide(w, total) for w in weight]
        return wt

    def get_distance_matrix(self, inputs):
        idx, sent_pred = inputs
        sent_gold = self.references[idx]
        sent_pred = sent_pred.split()
        sent_gold = sent_gold.split()
        m = len(sent_pred)
        n = len(sent_gold)

        d = np.ones((m, n))
        aligned_matrix = np.zeros((m, n))

        for i, word_pred in enumerate(sent_pred):
            for j, word_gold in enumerate(sent_gold):

                aligned_matrix[i][j] = self.align_score(i+1,j+1, sent_pred, sent_gold)

        mx_idx = np.argmax(aligned_matrix, axis=1)
        for i, id in enumerate(mx_idx):
            d[i][id] = 1.0 - (self.cosine_similarity(self.get_emb(sent_gold[id]), self.get_emb(sent_pred[i])) * np.exp(-1.0*self.pos_inf(i+1, id+1, m, n)))

        return d

    def get_weights(self, word, word_freq, doc_freq, N):
        '''
        get weights based on equation 9
        :return:
        '''
        #print ((np.log10(N/doc_freq[word])))
        if doc_freq[word]!=0:
            return (1+np.log10(word_freq[word])) * (np.log10(1 + (N/doc_freq[word])))
        else:
            return 0.0

    def get_word_frequencies(self, sents):
        '''
        return a dictionary with word and its count
        :param sents:
        :return:
        '''
        word_freq = defaultdict(float)
        for sent in sents:
            for w in sent.split():
                word_freq[w] += 1.0
        return word_freq

    def get_doc_frequencies(self, sents, word_freq):
        '''
        calculate doc frequencies
        :param sents:
        :param word_freq:
        :return:
        '''
        doc_freq = defaultdict(float)
        for w in word_freq:
            for sent in sents:
                if w in sent.split():
                    doc_freq[w] += 1.0
        return doc_freq

    def min_EMD_WORK(self, wN, wP, distance_matrix=None):
        '''
        Get EMD using PulP
        Get EMD using PulP
        :param wN:
        :param wP:
        :param distance_matrix:
        :return:
        '''
        d = distance_matrix
        wp, wn = wP, wN
        m, n = len(wp), len(wn)

        # Initialize the model
        model = pulp.LpProblem("Earth Mover Distance", pulp.LpMinimize)

        # Define Variable
        F = pulp.LpVariable.dicts("F", ((i, j) for i in range(m) for j in range(n)), lowBound=0, cat='Continuous')

        # Define objective function
        model += sum([d[i][j] * F[i, j] for i in range(m) for j in range(n)])  # Minimizing the WORK

        wpsum = np.sum(wp)
        wnsum = np.sum(wn)

        # constraint #1
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] >= 0

        # constraint #2
        for j, _ in enumerate(wn):
            for i, _ in enumerate(wp):
                model += F[i, j] <= wp[i]

        # constraint #3
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] <= wn[j]

        # constraint #4
        sumFij = min(wpsum, wnsum)
        model += pulp.lpSum(F) == sumFij

        try:
            model.solve()
            return pulp.value(model.objective), sumFij
        except Exception:
            print (wN, wP, distance_matrix)
            return 1.0, sumFij

    def min_EMD_WORK_parallel(self, wN, wP, distance_matrix=None):
        '''
        Get EMD using PulP
        :param wN:
        :param wP:
        :param distance_matrix:
        :return:
        '''
        #print (id)
        d = distance_matrix
        wp, wn = wP, wN
        m, n = len(wp), len(wn)

        # Initialize the model
        model = pulp.LpProblem("Earth Mover Distance", pulp.LpMinimize)

        # Define Variable
        F = pulp.LpVariable.dicts("F", ((i, j) for i in range(m) for j in range(n)), lowBound=0, cat='Continuous')

        # Define objective function
        model += sum([d[i][j] * F[i, j] for i in range(m) for j in range(n)])  # Minimizing the WORK

        wpsum = np.sum(wp)
        wnsum = np.sum(wn)

        # constraint #1
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] >= 0

        # constraint #2
        for j, _ in enumerate(wn):
            for i, _ in enumerate(wp):
                model += F[i, j] <= wp[i]

        # constraint #3
        for i, _ in enumerate(wp):
            for j, _ in enumerate(wn):
                model += F[i, j] <= wn[j]

        # constraint #4
        sumFij = min(wpsum, wnsum)
        model += pulp.lpSum(F) == sumFij

        try:
            model.solve()
            self.min_emd.append([pulp.value(model.objective),sumFij])
        except Exception as e:
            print (e)
            self.min_emd.append([0, sumFij])

    def we_wpi_multithread(self, sents_pred):
        '''
        calculate we_wpi multithreaded
        :param sents_gold:
        :param sents_pred:
        :return:
        '''
        assert(len(self.references)==len(sents_pred))
        print ('Getting WE WPI scores for predicted values')
        we_wpi_score = 0.0
        word_freq_trans = self.get_word_frequencies(sents_pred)

        doc_freq_trans = self.get_doc_frequencies(sents_pred, word_freq_trans)
        weights_trans = [[self.get_weights(w, word_freq_trans, doc_freq_trans, len(sents_pred)) for w in sent.split()] for sent in sents_pred]
        weights_trans = [self.calc_avg_weight(weight) for weight in weights_trans]
        dist_matrices = [self.get_distance_matrix(idxs) for idxs in enumerate(sents_pred)]

        start_time = time.clock()

        min_emd = [self.min_EMD_WORK(self.weights_ref[i], weights_trans[i], dist_matrices[i]) for i, s in tqdm(enumerate(weights_trans))]

        we_wpi_score = [(1.0 - np.divide(m_w, float(tot_fij))) if m_w is not None else 0 for m_w, tot_fij in min_emd]
        print ("Total time required: " + str((time.clock()-start_time)/60))
        return np.average(we_wpi_score)


if __name__ == "__main__":
    sentence_PRED = "jose mourinho is the current coach"
    sentence_GOLD = "manchester united 's manager is jose mourinho"

    #alist = ["he is the best player", "mourinho is the coach", "he plays as forward"]
    #blist = ["he is is", "he is is the", "he is is"]
    #alist = ['BUILDING is located in LOCATION , ISPARTOF , COUNTRY . The leader of COUNTRY is LEADERNAME and the language spoken there is English .']
    #blist = ['LEADERNAME is the leader of COUNTRY where the LANGUAGE language is spoken . LOCATION is located in the country and is where the ISPARTOF and BUILDING are found .']

    blist = ['Are there topics that you think should discuss world ?']
    #alist = ['Are there topics that you think should discuss world ?']
    alist = ['Are there topics you want to get the world talking about ?']
    #evaluator = Evaluate_we_wpi(embedding_path='/data2/mrony/GLMP/soccerbot_acl/vocab/cc.en.300.bin', vec_dim=300, references=alist)
    evaluator = Evaluate_we_wpi(embedding_path='/home/debanjan/acl_submissions/benchmark/soccerbot_acl/vocab/wiki.simple.bin', vec_dim=300, references=alist)
    print (evaluator.we_wpi_multithread(blist))
    
    #score = WE_WPI_score(sentence_GOLD, sentence_PRED)
    #print(score)
    #print (get_moses_multi_bleu(sentence_PRED, sentence_GOLD))
