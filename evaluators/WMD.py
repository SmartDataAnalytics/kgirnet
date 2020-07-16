from itertools import product
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import euclidean
import pulp
from unidecode import unidecode
from gensim.models import KeyedVectors
from ortools.linear_solver import pywraplp
import gensim


word_emb = KeyedVectors.load_word2vec_format("vocab/wiki-news-300d-1M.vec")

def tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}

def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets]) == second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets]) == first_sent_buckets[token1]

    if lpFile != None:
        prob.writeLP(lpFile)

    prob.solve()

    return prob





# def earthmover_distance(p1, p2):
#     dist1 = {x: count / len(p1) for (x, count) in Counter(p1).items()}
#     dist2 = {x: count / len(p2) for (x, count) in Counter(p2).items()}
#     solver = pywraplp.Solver('earthmover_distance', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
#
#     variables = dict()
#
#     # for each pile in dist1, the constraint that says all the dirt must leave this pile
#     dirt_leaving_constraints = defaultdict(lambda: 0)
#
#     # for each hole in dist2, the constraint that says this hole must be filled
#     dirt_filling_constraints = defaultdict(lambda: 0)
#
#     # the objective
#     objective = solver.Objective()
#     objective.SetMinimization()
#
#     for (x, dirt_at_x) in dist1.items():
#         for (y, capacity_of_y) in dist2.items():
#             amount_to_move_x_y = solver.NumVar(0, solver.infinity(), 'z_{%s, %s}' % (x, y))
#             variables[(x, y)] = amount_to_move_x_y
#             dirt_leaving_constraints[x] += amount_to_move_x_y
#             dirt_filling_constraints[y] += amount_to_move_x_y
#             objective.SetCoefficient(amount_to_move_x_y, euclidean(x, y))
#
#     for x, linear_combination in dirt_leaving_constraints.items():
#         solver.Add(linear_combination == dist1[x])
#
#     for y, linear_combination in dirt_filling_constraints.items():
#         solver.Add(linear_combination == dist2[y])
#
#     status = solver.Solve()
#     if status not in [solver.OPTIMAL, solver.FEASIBLE]:
#         raise Exception('Unable to find feasible solution')
#
#     return objective.Value()

sentence_GOLD = "Are there topics that you think should discuss world ?"
sentence_PRED = "Are there topics you want to get the world talking about ?"

sentence_GOLD = "Are there topics that you think should discuss world ?"
sentence_PRED = "This is not the kind of world your future generation want to see ?"

distance = word_mover_distance_probspec(unidecode(sentence_GOLD).split(), unidecode(sentence_PRED).split(), word_emb)

#print(distance)
print(pulp.value(distance.objective))


