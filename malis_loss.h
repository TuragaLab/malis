#ifndef MALIS_LOSS_H
#define MALIS_LOSS_H

void malis_loss_cpp(const int nVert, const int* seg,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               const int pos,
               int* nPairPerEdge);

#endif