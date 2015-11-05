from malisLoss import malis_loss
import numpy as np

seg = np.array([0, 1, 1, 1, 2, 2, 0, 5, 5, 5, 5],dtype=np.int32);
node1 = np.arange(seg.shape[0]-1,dtype=np.int32)
node2 = np.arange(1,seg.shape[0],dtype=np.int32)
edgeWeight = np.array([0, 1, 2, 0, 2, 0, 0, 1, 2, 3],dtype=np.float32);


nPairPos = malis_loss(seg, node1, node2, edgeWeight, 1)
nPairNeg = malis_loss(seg, node1, node2, edgeWeight, 0)
print nPairPos
print nPairNeg