#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	    const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const int& ind1, const int& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};

/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
void malis_loss_cpp(const int nVert, const int* seg,
               const int nEdge, const int* node1, const int* node2, const float* edgeWeight,
               const int pos,
               int* nPairPerEdge){


    /* Disjoint sets and sparse overlap vectors */
    vector<map<int,int> > overlap(nVert);
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i){
        dsets.make_set(i);
        if (0!=seg[i]) {
            overlap[i].insert(pair<int,int>(seg[i],1));
        }
    }

    /* Sort all the edges in increasing order of weight */
    std::vector< int > pqueue( nEdge );
    int j = 0;
    for ( int i = 0; i < nEdge; i++ ){
    	if ((node1[i] >= 0) && (node2[i] >= 0))
	        pqueue[ j++ ] = i;
    }
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edgeWeight ) );


    /* Start MST */
    int minEdge;
    int set1, set2;
    int nPair = 0;
    map<int,int>::iterator it1, it2;

    /* Start Kruskal's */
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
        minEdge = pqueue[i];

        set1 = dsets.find_set(node1[minEdge]);
        set2 = dsets.find_set(node2[minEdge]);

        if (set1!=set2){
            dsets.link(set1, set2);

            /* compute the number of pairs merged by this MST edge */
            for (it1 = overlap[set1].begin();
                    it1 != overlap[set1].end(); ++it1) {
                for (it2 = overlap[set2].begin();
                        it2 != overlap[set2].end(); ++it2) {

                    nPair = it1->second * it2->second;

                    if (pos && (it1->first == it2->first)) {
                        nPairPerEdge[minEdge] += nPair;
                    } else if ((!pos) && (it1->first != it2->first)) {
                        nPairPerEdge[minEdge] += nPair;
                    }
                }
            }

            /* move the pixel bags of the non-representative to the representative */
            if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
                swap(set1,set2);

            it2 = overlap[set2].begin();
            while (it2 != overlap[set2].end()) {
                it1 = overlap[set1].find(it2->first);
                if (it1 == overlap[set1].end()) {
                    overlap[set1].insert(pair<int,int>(it2->first,it2->second));
                } else {
                    it1->second += it2->second;
                }
                overlap[set2].erase(it2++);
            }
        } // end link

    } // end while
}


void connected_components_cpp(const int nVert,
               const int nEdge, const int* node1, const int* node2,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(nVert);
    vector<int> parent(nVert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<nVert; ++i)
        dsets.make_set(i);

    /* union */
    for (int i = 0; i < nEdge; ++i )
         // check bounds to make sure the nodes are valid
        if ((node1[i]>=0) && (node1[i]<nVert) && (node2[i]>=0) && (node2[i]<nVert))
            dsets.union_set(node1[i],node2[i]);

    /* find */
    for (int i = 0; i < nVert; ++i)
        seg[i] = dsets.find_set(i);
}
