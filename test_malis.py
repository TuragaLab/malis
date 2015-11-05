import numpy as np
import malis as m

segTrue = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5],dtype=np.int32);
node1 = np.arange(segTrue.shape[0]-1,dtype=np.int32)
node2 = np.arange(1,segTrue.shape[0],dtype=np.int32)
nVert = segTrue.shape[0]
edgeWeight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 3],dtype=np.float32);
print segTrue
print edgeWeight

nPairPos = m.malis_loss(segTrue, node1, node2, edgeWeight, 1)
nPairNeg = m.malis_loss(segTrue, node1, node2, edgeWeight, 0)
print np.vstack((nPairPos,nPairNeg))
# print nPairNeg

idxkeep = (edgeWeight > 0).astype(np.int32)
cc = m.connected_components(nVert,node1,node2,idxkeep)
print cc