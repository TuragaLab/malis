import numpy as np
import malis as m
import h5py
np.set_printoptions(precision=4)

segTrue = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5],dtype=np.int32);
node1 = np.arange(segTrue.shape[0]-1,dtype=np.int32)
node2 = np.arange(1,segTrue.shape[0],dtype=np.int32)
nVert = segTrue.shape[0]
edgeWeight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 2.5],dtype=np.float32);
edgeWeight = edgeWeight/edgeWeight.max()
print segTrue
print edgeWeight

nPairPos = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 1)
nPairNeg = m.malis_loss_weights(segTrue, node1, node2, edgeWeight, 0)
print np.vstack((nPairPos,nPairNeg))
# print nPairNeg

idxkeep = (edgeWeight > 0).astype(np.int32)
cc = m.connected_components(nVert,node1,node2,idxkeep)
print cc


# node1, node2 = m.nodelist_like((2,3,4),-np.eye(3))
# print node1
# print node2