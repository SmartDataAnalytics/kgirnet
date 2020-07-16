import pulp
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

def EMD_WORK(P,Q, distance_matrix= None):
     m = len(P)
     n = len(Q)

     d = np.zeros((m,n))
     for i in range(m):
          for j in range(n):
               x1,y1 = P[i]
               x2,y2 = Q[j]
               d[i][j] = euclidean_distances([[x1,y1]],[[x2,y2]])[0][0]
     #print(d)

     wp = [0.6,0.6,0.6,0.6]
     wn = [0.8,0.8,0.8]

     # Initialize the model
     model = pulp.LpProblem("Earth Mover distance", pulp.LpMinimize)

     # Define Variable
     F = pulp.LpVariable.dicts("F", ((i,j) for i in range(m) for j in range(n)), lowBound=0, cat='Continuous')

     # Define objective function
     model += pulp.lpSum([d[i][j] *F[i,j] for i in range(m) for j in range(n)])     #Minimizing the WORK

     wpsum = sum(wp)
     wnsum = sum(wn)

     #constraint #1
     for i,_ in enumerate(wp):
          for j,_ in enumerate(wn):
               model += F[i,j]>=0

     #constraint #2
     for j,_ in enumerate(wn):
          for i, _ in enumerate(wp):
               model += F[i,j]<=wp[i]

     #constraint #3
     for i,_ in enumerate(wp):
          for j, _ in enumerate(wn):
               model += F[i,j]<=wn[j]

     #constraint #4
     sumFij = min(wpsum,wnsum)
     model += pulp.lpSum(F)==sumFij


     model.solve()
     print(pulp.LpStatus[model.status])
     print(model.objective)
     print(pulp.value(model.objective))

     for i,_ in enumerate(wp):
          for j,_ in enumerate(wn):
               print("F"+str(i+1)+str(j+1)+":"+str(F[i,j].value()))


P = [(1,5),(5,5),(1,1),(5,1)]
Q = [(2,3),(4,3),(3,2)]

distance_matrix = np.zeros((4,3))
print(EMD_WORK(P,Q,distance_matrix))