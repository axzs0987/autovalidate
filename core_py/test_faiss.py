import numpy as np
import faiss
 
d = 2048                          # dimension
nb = 7030                     # database size
nq = 10                       # nb of queries
 
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
 
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
 
index = faiss.IndexFlatL2(d)
print(index.is_trained)
 
index.add(xb)
print(index.ntotal)
 
k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print("I: ",I)
print("D: ",D)