#ifndef MALIS_CPP_H
#define MALIS_CPP_H

void connected_components_cpp(const int nVert,
               const int nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
               uint64_t* seg);

void malis_loss_weights_cpp(const int nVert, const uint64_t* seg,
               const int nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
               const int pos,
               uint64_t* nPairPerEdge);

void marker_watershed_cpp(const int nVert, const uint64_t* marker,
               const int nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
               uint64_t* seg);
#endif