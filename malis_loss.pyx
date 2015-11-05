import numpy as np
cimport numpy as np

cdef extern from "malis_loss.h":
    void malis_loss_cpp(const int nVert, const int* seg,
                   const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
                   const int pos,
                   int* nPairPerEdge);

def malis_loss(np.ndarray[np.int32_t,ndim=1] seg,
                np.ndarray[np.int32_t,ndim=1] node1,
                np.ndarray[np.int32_t,ndim=1] node2,
                np.ndarray[np.float32_t,ndim=1] edgeWeight,
                np.int pos):
    cdef nVert = seg.shape[0]
    seg = np.ascontiguousarray(seg,dtype=np.int32)
    cdef nEdge = edgeWeight.shape[0]
    cdef np.ndarray[np.int32_t,ndim=1] nPairPerEdge = np.zeros(edgeWeight.shape[0],dtype=np.int32)
    malis_loss_cpp(nVert, <int*> &seg[0],
                   nEdge, <int*> &node1[0], <int*> &node2[0], <float*> &edgeWeight[0],
                   pos,
                   <int*> &nPairPerEdge[0]);
    return nPairPerEdge