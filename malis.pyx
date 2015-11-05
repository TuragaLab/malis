import numpy as np
cimport numpy as np

cdef extern from "malis_cpp.h":
    void malis_loss_cpp(const int nVert, const int* segTrue,
                   const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
                   const int pos,
                   int* nPairPerEdge);
    void connected_components_cpp(const int nVert,
                   const int nEdge, const int* node1, const int* node2,
                   int* seg);

def malis_loss(np.ndarray[np.int32_t,ndim=1] segTrue,
                np.ndarray[np.int32_t,ndim=1] node1,
                np.ndarray[np.int32_t,ndim=1] node2,
                np.ndarray[np.float32_t,ndim=1] edgeWeight,
                np.int pos):
    cdef int nVert = segTrue.shape[0]
    cdef int nEdge = node1.shape[0]
    segTrue = np.ascontiguousarray(segTrue)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edgeWeight = np.ascontiguousarray(edgeWeight)
    cdef np.ndarray[np.int32_t,ndim=1] nPairPerEdge = np.zeros(edgeWeight.shape[0],dtype=np.int32)
    malis_loss_cpp(nVert, <int*> &segTrue[0],
                   nEdge, <int*> &node1[0], <int*> &node2[0], <float*> &edgeWeight[0],
                   pos,
                   <int*> &nPairPerEdge[0]);
    return nPairPerEdge

def connected_components(np.int nVert,
                         np.ndarray[np.int32_t,ndim=1] node1,
                         np.ndarray[np.int32_t,ndim=1] node2):
    cdef int nEdge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    cdef np.ndarray[np.int32_t,ndim=1] seg = np.zeros(nVert,dtype=np.int32)
    connected_components_cpp(nVert,
                             nEdge, <int*> &node1[0], <int*> &node2[0],
                             <int*> &seg[0]);
    return seg


def nodelist_like(aff,nhood):
    # constructs the node lists corresponding to the edge list representation of an affinity graph
    # assume affinity graph is represented as:
    # aff.shape = (edges, z, y, x)
    nodes = np.arange(np.prod(aff.shape[1:]),dtype=np.int32).reshape(aff.shape[1:])
    node1 = np.tile(nodes,(aff.shape[0],1,1,1))
    node2 = np.zeros_like(aff,dtype=np.int32).fill(-1)

    # for e in range(nhood.shape[0]):
    #     node2[]

    return (node1, node2, aff.ravel())