#include "mex.h"
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
		bool operator() (const mwIndex& ind1, const mwIndex& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};



/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

	/* input arrays */
    // 4d connectivity graph [y * x * z * #edges]
    const mxArray*	conn				= prhs[0];
	const mwSize	conn_num_dims		= mxGetNumberOfDimensions(conn);
	const mwSize*	conn_dims			= mxGetDimensions(conn);
	const mwSize	conn_num_elements	= mxGetNumberOfElements(conn);
	const float*	conn_data			= (const float*)mxGetData(conn);
    // graph neighborhood descriptor [3 * #edges]
	const mxArray*	nhood				= prhs[1];
	const mwSize	nhood_num_dims		= mxGetNumberOfDimensions(nhood);
	const mwSize*	nhood_dims			= mxGetDimensions(nhood);
	const double*	nhood_data			= (const double*)mxGetData(nhood);
    // true target segmentation [y * x * z]
    const mxArray*	seg					= prhs[2];
	const mwSize	seg_num_dims		= mxGetNumberOfDimensions(seg);
	const mwSize*	seg_dims			= mxGetDimensions(seg);
	const mwSize	seg_num_elements	= mxGetNumberOfElements(seg);
	const uint16_t*	seg_data			= (const uint16_t*)mxGetData(seg);
    // sq-sq loss margin [0.3]
    const double    margin              = (const double)mxGetScalar(prhs[3]);
    // is this a positive example pass [true] or a negative example pass [false] ?
    const bool      pos                 = mxIsLogicalScalarTrue(prhs[4]);

	if (!mxIsSingle(conn)){
		mexErrMsgTxt("Conn array must be floats (singles)");
	}
	if (nhood_num_dims != 2) {
		mexErrMsgTxt("wrong size for nhood");
	}
	if ((nhood_dims[1] != (conn_num_dims-1))
		|| (nhood_dims[0] != conn_dims[conn_num_dims-1])){
		mexErrMsgTxt("nhood and conn dimensions don't match");
	}
	if (!mxIsUint16(seg)){
		mexErrMsgTxt("seg array must be uint16");
	}

    /* Matlab Notes:
     * mwSize and mwIndex are functionally equivalent to unsigned ints.
     * mxArray is an n-d array c-style "container" basically containing
     * a linear array with meta-data describing the dimension sizes, etc
     */

	/* Output arrays */
    // the derivative of the MALIS-SqSq loss function
    // (times the derivative of the logistic activation function) [y * x * z * #edges]
    plhs[0] = mxCreateNumericArray(conn_num_dims,conn_dims,mxSINGLE_CLASS,mxREAL);
    mxArray* dloss = plhs[0];
    float* dloss_data = (float*)mxGetData(dloss);

	/* Cache for speed to access neighbors */
	mwSize nVert = 1;
	for (mwIndex i=0; i<conn_num_dims-1; ++i)
		nVert = nVert*conn_dims[i];

	vector<mwSize> prodDims(conn_num_dims-1); prodDims[0] = 1;
	for (mwIndex i=1; i<conn_num_dims-1; ++i)
		prodDims[i] = prodDims[i-1]*conn_dims[i-1];

    /* convert n-d offset vectors into linear array offset scalars */
	vector<int32_t> nHood(nhood_dims[0]);
	for (mwIndex i=0; i<nhood_dims[0]; ++i) {
		nHood[i] = 0;
		for (mwIndex j=0; j<nhood_dims[1]; ++j) {
			nHood[i] += (int32_t)nhood_data[i+j*nhood_dims[0]] * prodDims[j];
		}
	}

	/* Disjoint sets and sparse overlap vectors */
	vector<map<mwIndex,mwIndex> > overlap(nVert);
	vector<mwIndex> rank(nVert);
	vector<mwIndex> parent(nVert);
	map<mwIndex,mwSize> segSizes;
	mwSize nLabeledVert=0;
    mwSize nPairPos=0;
	boost::disjoint_sets<mwIndex*, mwIndex*> dsets(&rank[0],&parent[0]);
	for (mwIndex i=0; i<nVert; ++i){
		dsets.make_set(i);
		if (0!=seg_data[i]) {
			overlap[i].insert(pair<mwIndex,mwIndex>(seg_data[i],1));
			++nLabeledVert;
            ++segSizes[seg_data[i]];
            nPairPos += (segSizes[seg_data[i]]-1);
		}
	}
	mwSize nPairTot = (nLabeledVert*(nLabeledVert-1))/2;
    mwSize nPairNeg = nPairTot - nPairPos;
    mwSize nPairNorm;
    if (pos) {nPairNorm = nPairPos;} else {nPairNorm = nPairNeg;}

	/* Sort all the edges in increasing order of weight */
	std::vector< mwIndex > pqueue( static_cast< mwSize >(3) *
								   ( conn_dims[0]-1 ) *
								   ( conn_dims[1]-1 ) *
								   ( conn_dims[2]-1 ));
	mwIndex j = 0;
	for ( mwIndex d = 0, i = 0; d < conn_dims[3]; ++d )
		for ( mwIndex z = 0; z < conn_dims[2]; ++z )
			for ( mwIndex y = 0; y < conn_dims[1]; ++y )
				for ( mwIndex x = 0; x < conn_dims[0]; ++x, ++i )
				{
					if ( x > 0 && y > 0 && z > 0 )
						pqueue[ j++ ] = i;
				}
	sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( conn_data ) );

	/* Start MST */
	mwIndex minEdge;
	mwIndex e, v1, v2;
	mwIndex set1, set2, tmp;
    mwSize nPair = 0;
	double loss=0, dl=0;
    mwSize nPairIncorrect = 0;
	map<mwIndex,mwIndex>::iterator it1, it2;

    /* Start Kruskal's */
    for ( mwIndex i = 0; i < pqueue.size(); ++i ) {
		minEdge = pqueue[i];
		e = minEdge/nVert; v1 = minEdge%nVert; v2 = v1+nHood[e];

		set1 = dsets.find_set(v1);
		set2 = dsets.find_set(v2);
		if (set1!=set2){
			dsets.link(set1, set2);

			/* compute the dloss for this MST edge */
			for (it1 = overlap[set1].begin();
					it1 != overlap[set1].end(); ++it1) {
				for (it2 = overlap[set2].begin();
						it2 != overlap[set2].end(); ++it2) {

                    nPair = it1->second * it2->second;

					if (pos && (it1->first == it2->first)) {
                        // +ve example pairs
                        // Sq-Sq loss is used here
                        dl = max(0.0,0.5+margin-conn_data[minEdge]);
                        loss += 0.5*dl*dl*nPair;
                        dloss_data[minEdge] += dl*nPair;
                        if (conn_data[minEdge] <= 0.5) { // an error
                            nPairIncorrect += nPair;
                        }

					} else if ((!pos) && (it1->first != it2->first)) {
                        // -ve example pairs
                        // Sq-Sq loss is used here
						dl = -max(0.0,conn_data[minEdge]-0.5+margin);
                        loss += 0.5*dl*dl*nPair;
                        dloss_data[minEdge] += dl*nPair;
                        if (conn_data[minEdge] > 0.5) { // an error
                            nPairIncorrect += nPair;
                        }
					}
				}
			}
            dloss_data[minEdge] /= nPairNorm;
            /* HARD-CODED ALERT!!
             * The derivative of the activation function is also multiplied here.
             * Assumes the logistic nonlinear activation function.
             */
            dloss_data[minEdge] *= conn_data[minEdge]*(1-conn_data[minEdge]); // DSigmoid

			/* move the pixel bags of the non-representative to the representative */
			if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
				swap(set1,set2);

			it2 = overlap[set2].begin();
			while (it2 != overlap[set2].end()) {
				it1 = overlap[set1].find(it2->first);
				if (it1 == overlap[set1].end()) {
					overlap[set1].insert(pair<mwIndex,mwIndex>(it2->first,it2->second));
				} else {
					it1->second += it2->second;
				}
				overlap[set2].erase(it2++);
			}
		} // end link

	} // end while

    /* Return items */
    double classerr, randIndex;
    if (nlhs > 1) {
        loss /= nPairNorm;
        plhs[1] = mxCreateDoubleScalar(loss);
    }
    if (nlhs > 2) {
        classerr = (double)nPairIncorrect / (double)nPairNorm;
        plhs[2] = mxCreateDoubleScalar(classerr);
    }
    if (nlhs > 3) {
        randIndex = 1.0 - ((double)nPairIncorrect / (double)nPairNorm);
        plhs[3] = mxCreateDoubleScalar(randIndex);
    }
}
