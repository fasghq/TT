from jax import grad
import jax.numpy as np
import numpy as onp
from jax.numpy import linalg as la
import math


# input tensor A
'''
A = np.array([[1/2, 1/3, 1/4], [1/3, 1/4, 1/5], [1/4, 1/5, 1/6]])
A = np.arange(24)
B = A.reshape(4, 3, 2)
A = B
'''

'''
A = np.ones(24)
A = A.reshape(4, 3, 2)
'''

A = []
for i in range(100):
  for j in range(100):
    for q in range(100):
      A.append(1 / (i + j + q + 3))
A = np.asarray(A)
A = A.reshape(100, 100, 100)

d = len(A.shape)
N = np.size(A)
n = A.shape

eps = 1e-12 # accuracy
delta = (eps/math.sqrt(d-1)) * la.norm(A) # cutting param

C = A # tmp tensor

G = [] # tt-cores
r = [] # tt-ranks
r.append(1)

for k in range(1, d):
  C = np.reshape(C, (r[k-1] * n[k-1], int(N / (r[k-1] * n[k-1]))))
  
  # calc low-rank approximation
  u, s, v = la.svd(C)
  sum = 0 
  nsize = np.size(s)
  rres = np.size(s)
  for rk in range(0, nsize):
    for m in range(rk+1, nsize):
      sum = sum + (s[m] ** 2)
    if (sum <= (eps ** 2) * la.norm(A)) and (rres > rk):
      rres = rk + 1 
    sum = 0
  r.append(rres) 

  G.append(np.reshape(u[:, :r[k]], (r[k-1], n[k-1], r[k])))
  s = np.diag(s)
  C = np.dot(s[:r[k], :r[k]], v[:r[k], :])
  N = (N * r[k]) / (n[k-1] * r[k-1])
G.append(C) 

for X in G:
  print(X, "\n")
print(r)


