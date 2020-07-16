import numpy as np

# def wer(r, h):
#     "CODE TAKEN FROM:   https://martin-thoma.com/word-error-rate-calculation/"
#     """
#     Calculation of WER with Levenshtein distance.
#
#     Works only for iterables up to 254 elements (uint8).
#     O(nm) time ans space complexity.
#
#     Parameters
#     ----------
#     r : list
#     h : list
#
#     Returns
#     -------
#     int
#
#     Examples
#     --------
#     >>> wer("who is there".split(), "is there".split())
#     1
#     >>> wer("who is there".split(), "".split())
#     3
#     >>> wer("".split(), "who is there".split())
#     3
#     """
#     # initialisation
#     import numpy
#     d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
#     d = d.reshape((len(r)+1, len(h)+1))
#     for i in range(len(r)+1):
#         for j in range(len(h)+1):
#             if i == 0:
#                 d[0][j] = j
#             elif j == 0:
#                 d[i][0] = i
#
#     # computation
#     for i in range(1, len(r)+1):
#         for j in range(1, len(h)+1):
#             if r[i-1] == h[j-1]:
#                 d[i][j] = d[i-1][j-1]
#             else:
#                 substitution = d[i-1][j-1] + 1
#                 insertion    = d[i][j-1] + 1
#                 deletion     = d[i-1][j] + 1
#                 d[i][j] = min(substitution, insertion, deletion)
#
#     return d[len(r)][len(h)]
#
#

def wer_score(hyp, ref, print_matrix=False):
  N = len(hyp)
  M = len(ref)
  L = np.zeros((N,M))
  for i in range(0, N):
    for j in range(0, M):
      if min(i,j) == 0:
        L[i,j] = max(i,j)
      else:
        deletion = L[i-1,j] + 1
        insertion = L[i,j-1] + 1
        sub = 1 if hyp[i] != ref[j] else 0
        substitution = L[i-1,j-1] + sub
        L[i,j] = min(deletion, min(insertion, substitution))
        # print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
  if print_matrix:
    print("WER matrix ({}x{}): ".format(N, M))
    print(L)
  return int(L[N-1, M-1])