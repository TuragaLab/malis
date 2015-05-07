#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
#include <boost/python/tuple.hpp>
#include <boost/python.hpp>

using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const size_t& ind1, const size_t& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};



/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
//void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){

boost::python::tuple malisLoss(
    vigra::NumpyArray<4, float> conn,
    vigra::NumpyArray<2, double> nhood,
    vigra::NumpyArray<3, vigra::UInt16> seg,
    vigra::NumpyArray<4, float> dloss,
    double margin,
    bool pos,
    bool neg
) {
#if 0
    if(conn(4,8,20,2) != 42) {
        throw std::runtime_error("wrong xxx");
    }
    else {
        //C-order
        float* magicEntry = conn.data() + 4*conn.shape(1)*conn.shape(2)*conn.shape(3) + 8*conn.shape(2)*conn.shape(3) + 20*conn.shape(3) + 2;
        
        //FORTRAN-order
        float* magicEntry2 = conn.data() + 2*conn.shape(0)*conn.shape(1)*conn.shape(2) + 20*conn.shape(0)*conn.shape(1) + 8*conn.shape(0) + 4;
        if(*magicEntry != 42) {
            std::cout << "is not c-order" << std::endl;
        }
        if(*magicEntry2 != 42) {
            std::cout << "is not fortran-order" << std::endl;
        }
        std::cout << "strides = " << conn.isUnstrided() << std::endl;
        std::cout << "stride = [" << conn.stride(0) << ", " << conn.stride(1) << ", " << conn.stride(2) << ", " << conn.stride(3) << std::endl;
    }
#endif
    
	/* input arrays */
    // 4d connectivity graph [y * x * z * #edges]
#if 0
    const mxArray*	conn				= prhs[0];
	const size_t	conn_num_dims		= mxGetNumberOfDimensions(nhood); 
	const size_t*	conn_dims			= mxGetDimensions(conn);
	const size_t	conn_num_elements	= mxGetNumberOfElements(conn);
	const float*	conn_data			= (const float*)mxGetData(conn);
#else
    const size_t conn_num_dims = 4;
    vigra::NumpyArray<4, float>::size_type conn_dims = conn.shape();
    const size_t conn_num_elements = conn.size();
    const float* conn_data = (const float*)conn.data();
#endif

    // graph neighborhood descriptor [3 * #edges]
#if 0
	const mxArray*	nhood				= prhs[1];
	const size_t	nhood_num_dims		= mxGetNumberOfDimensions(nhood);
	const size_t*	nhood_dims			= mxGetDimensions(nhood);
	const double*	nhood_data			= (const double*)mxGetData(nhood);
#else
    const size_t nhood_num_dims = 2;
    vigra::NumpyArray<2, double>::size_type nhood_dims = nhood.shape();
    const size_t nhood_num_elements = nhood.size();
    const double* nhood_data = (const double*)nhood.data();
#endif

    // true target segmentation [y * x * z]
#if 0
    const mxArray*	seg					= prhs[2];
	const size_t	seg_num_dims		= mxGetNumberOfDimensions(seg);
	const size_t*	seg_dims			= mxGetDimensions(seg);
	const size_t	seg_num_elements	= mxGetNumberOfElements(seg);
	const uint16_t*	seg_data			= (const uint16_t*)mxGetData(seg);
#else
    const size_t seg_num_dims = 3;
    vigra::NumpyArray<3, vigra::UInt16>::size_type seg_dims = seg.shape();
    const size_t seg_num_elements = seg.size();
    const vigra::UInt16* seg_data = (const vigra::UInt16*)seg.data();
#endif

#if 0
    // sq-sq loss margin [0.3]
    const double    margin              = (const double)mxGetScalar(prhs[3]);
    // is this a positive example pass [true] or a negative example pass [false] ?
    const bool      pos                 = mxIsLogicalScalarTrue(prhs[4]);
#endif

#if 0
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
#else
	if ((nhood_dims[1] != (conn_num_dims-1))
		|| (nhood_dims[0] != conn_dims[conn_num_dims-1])){
        throw std::runtime_error("nhood and conn dimensions don't match");
	}
#endif

#if 0
	if (!mxIsUint16(seg)){
		mexErrMsgTxt("seg array must be uint16");
	}
#endif

    /* Matlab Notes:
     * size_t and size_t are functionally equivalent to unsigned ints.
     * mxArray is an n-d array c-style "container" basically containing
     * a linear array with meta-data describing the dimension sizes, etc
     */

	/* Output arrays */
    // the derivative of the MALIS-SqSq loss function
    // (times the derivative of the logistic activation function) [y * x * z * #edges]
    
#if 0    
    //plhs[0] = mxCreateNumericArray(conn_num_dims,conn_dims,mxSINGLE_CLASS,mxREAL);
    //mxArray* dloss = plhs[0];
    //float* dloss_data = (float*)mxGetData(dloss);
#endif
    
    float* dloss_data = dloss.data();

	/* Cache for speed to access neighbors */
	size_t nVert = 1;
	for (size_t i=0; i<conn_num_dims-1; ++i)
		nVert = nVert*conn_dims[i];

	vector<size_t> prodDims(conn_num_dims-1); prodDims[0] = 1;
	for (size_t i=1; i<conn_num_dims-1; ++i)
		prodDims[i] = prodDims[i-1]*conn_dims[i-1];

    /* convert n-d offset vectors into linear array offset scalars */
	vector<int32_t> nHood(nhood_dims[0]);
	for (size_t i=0; i<nhood_dims[0]; ++i) {
		nHood[i] = 0;
		for (size_t j=0; j<nhood_dims[1]; ++j) {
			nHood[i] += (int32_t)nhood_data[i+j*nhood_dims[0]] * prodDims[j];
		}
		//std::cout << "nHood[" << i << "] = " << nHood[i] << std::endl;
	}

	/* Disjoint sets and sparse overlap vectors */
	vector<map<size_t,size_t> > overlap(nVert);
	vector<size_t> rank(nVert);
	vector<size_t> parent(nVert);
	map<size_t,size_t> segSizes;
	size_t nLabeledVert=0;
    size_t nPairPos=0;
	boost::disjoint_sets<size_t*, size_t*> dsets(&rank[0],&parent[0]);
	for (size_t i=0; i<nVert; ++i){
		dsets.make_set(i);
		if (0!=seg_data[i]) {
			overlap[i].insert(pair<size_t,size_t>(seg_data[i],1));
			++nLabeledVert;
            ++segSizes[seg_data[i]];
            nPairPos += (segSizes[seg_data[i]]-1);
		}
	}
	size_t nPairTot = (nLabeledVert*(nLabeledVert-1))/2;
    size_t nPairNeg = nPairTot - nPairPos;
    size_t nPairNorm = 0;
    if (pos)
    	nPairNorm += nPairPos;
    if (neg)
    	nPairNorm = nPairNeg;

	/* Sort all the edges in increasing order of weight */
	std::vector< size_t > pqueue( static_cast< size_t >(3) *
								   ( conn_dims[0]-1 ) *
								   ( conn_dims[1]-1 ) *
								   ( conn_dims[2]-1 ));
	size_t j = 0;
	for ( size_t d = 0, i = 0; d < conn_dims[3]; ++d )
		for ( size_t z = 0; z < conn_dims[2]; ++z )
			for ( size_t y = 0; y < conn_dims[1]; ++y )
				for ( size_t x = 0; x < conn_dims[0]; ++x, ++i )
				{
					if ( x > 0 && y > 0 && z > 0 )
						pqueue[ j++ ] = i;
				}
	sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( conn_data ) );

	/* Start MST */
	size_t minEdge;
	size_t e, v1, v2;
	size_t set1, set2, tmp;
    size_t nPair = 0;
	double loss=0, dl=0;
    size_t nPairIncorrect = 0;
	map<size_t,size_t>::iterator it1, it2;

    /* Start Kruskal's */
    for ( size_t i = 0; i < pqueue.size(); ++i ) {
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

					}
					if (neg && (it1->first != it2->first)) {
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
            //dloss_data[minEdge] *= conn_data[minEdge]*(1-conn_data[minEdge]); // DSigmoid

			/* move the pixel bags of the non-representative to the representative */
			if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
				swap(set1,set2);

			it2 = overlap[set2].begin();
			while (it2 != overlap[set2].end()) {
				it1 = overlap[set1].find(it2->first);
				if (it1 == overlap[set1].end()) {
					overlap[set1].insert(pair<size_t,size_t>(it2->first,it2->second));
				} else {
					it1->second += it2->second;
				}
				overlap[set2].erase(it2++);
			}
		} // end link

	} // end while

    /* Return items */
    double classerr, randIndex;
#if 0
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
#else
    loss /= nPairNorm;
    classerr = (double)nPairIncorrect / (double)nPairNorm;
    randIndex = 1.0 - ((double)nPairIncorrect / (double)nPairNorm);
    
    return boost::python::make_tuple(dloss, loss, classerr, randIndex);
#endif
     
    
}

BOOST_PYTHON_MODULE_INIT(pymalis) {
    vigra::import_vigranumpy();
    boost::python::def("malisLoss", 
        vigra::registerConverters(&malisLoss)
    );
}



