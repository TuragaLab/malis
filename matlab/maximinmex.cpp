

#include "mex.h"
#include <iostream>
#include <cstdlib>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
using namespace std;

// zero-based sub2ind
mwSize sub2ind(
		const mwSize * sub,
		const mwSize num_dims,
		const mwSize * dims
		)
{
	mwSize ind = 0;
	mwSize prod = 1;
	for (mwSize d=0; d<num_dims; d++) {
		ind += sub[d] * prod;
		prod *= dims[d];
	}
	return ind;
}

mwSize dblsub2ind(
		const double * sub,
		const mwSize num_dims,
		const mwSize * dims
		)
{
	mwSize ind = 0;
	mwSize prod = 1;
	for (mwSize d=0; d<num_dims; d++) {
		ind += (const mwSize)sub[d] * prod;
		prod *= dims[d];
	}
	return ind;
}

// zero-based ind2sub
void ind2sub(
		mwSize ind,
		const mwSize num_dims,
		const mwSize * dims,
		mwSize * sub
		)
{
	for (mwSize d=0; d<num_dims; d++) {
		sub[d] = (ind % dims[d]);
		ind /= dims[d];
	}
	return;
}



class mycomp{
    const float * conn_data;
    public:
        mycomp(const float * conn_data_param){
            conn_data = conn_data_param;
        }
        bool operator() (const mwSize& ind1, const mwSize& ind2) const {
            return conn_data[ind1]<conn_data[ind2];
        }
};
    
//MAXIMUM spanning tree
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    const mxArray * conn = prhs[0];
	const mwSize conn_num_dims = mxGetNumberOfDimensions(conn);
	const mwSize * conn_dims = mxGetDimensions(conn);
	const mwSize conn_num_elements = mxGetNumberOfElements(conn);
	const float * conn_data =(const float *)mxGetData(conn);
	const mxArray * nhood = prhs[1];
	const mwSize nhood_num_dims = mxGetNumberOfDimensions(nhood);
	const mwSize * nhood_dims = mxGetDimensions(nhood);
	const double * nhood_data = (const double *)mxGetData(nhood);
    const mxArray * p1 = prhs[2];
    const mxArray * p2 = prhs[3];
    const double * p1_data=(const double *) mxGetData(p1);
    const double * p2_data=(const double *) mxGetData(p2);
    const mwSize * p1_dims=mxGetDimensions(p1);
    const mwSize * p2_dims=mxGetDimensions(p2);
    const mwSize p1_num_dims=mxGetNumberOfDimensions(p1);
    const mwSize p2_num_dims=mxGetNumberOfDimensions(p2);

    if (p1_num_dims!=2 || p2_num_dims!=2 || p1_dims[0]!=conn_num_dims-1 || p2_dims[0]!=conn_num_dims-1 || p1_dims[1]!=p2_dims[1]){
        mexErrMsgTxt("pixel sizes are wrong");
    }
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

    /*mxArray * mst;
    mwSize mst_num_dims=conn_num_dims;
    mwSize mst_dims[mst_num_dims];
    for (mwSize i=0; i<mst_num_dims; i++){
        mst_dims[i]=conn_dims[i];
    }
    plhs[0]=mxCreateLogicalArray(mst_num_dims, mst_dims);
    mst=plhs[0];
    mxLogical * mst_data=(mxLogical *)mxGetData(mst);
    mwSize mst_num_elements=mxGetNumberOfElements(mst);
    for (mwSize i=0; i<mst_num_elements; i++){
        mst_data[i]=0;
    }*/

    mxArray * conn_edge;
    mwSize conn_edge_num_dims=2;
    mwSize conn_edge_dims[2];
    conn_edge_dims[0]=conn_num_dims;
    conn_edge_dims[1]=p1_dims[1];
    plhs[0]=mxCreateNumericArray(conn_edge_num_dims,conn_edge_dims,mxDOUBLE_CLASS,mxREAL);
    conn_edge=plhs[0];
    double * conn_edge_data=(double *)mxGetData(conn_edge);

    mwSize num_vertices=conn_dims[0]*conn_dims[1]*conn_dims[2];
    std::vector<mwSize> rank(num_vertices);
    std::vector<mwSize> parent(num_vertices);
    boost::disjoint_sets<mwSize*, mwSize*> dsets(&rank[0],&parent[0]);
    for (mwSize i=0; i<num_vertices; i++){
        dsets.make_set(i);
    }

    std::priority_queue <mwSize, vector<mwSize>, mycomp > pqueue (conn_data);

    for (mwSize i=0; i<conn_num_elements; i++){
        pqueue.push(i);
        /*cout << conn_data[i] << " " << i << endl;
        cout << "size is " << pqueue.size() << endl;
        cout << "top is " << pqueue.top() << endl << endl;*/
    }

    bool stop=false;
    mwSize p1_ind[p1_dims[1]];
    mwSize p2_ind[p1_dims[1]];
	mwSize cur_edge;
    mwSize edge_array[conn_num_dims];

	// switch from 1-base to 0-base
	mwSize p1_sub[(conn_num_dims-1)*p1_dims[1]], p2_sub[(conn_num_dims-1)*p1_dims[1]];
	for (mwSize i=0;i<conn_num_dims-1;i++){
        for (mwSize j=0; j<p1_dims[1]; j++){
		    p1_sub[i+j*(conn_num_dims-1)] = (mwSize) p1_data[i+j*p1_dims[0]]-1;
		    p2_sub[i+j*(conn_num_dims-1)] = (mwSize) p2_data[i+j*p1_dims[0]]-1;
        }
	}


    for (mwSize i=0; i<p1_dims[1]; i++){
        mwSize sub[conn_num_dims-1];
        for (mwSize j=0; j<conn_num_dims-1; j++){
            sub[j]=p1_sub[i*(conn_num_dims-1)+j];

        }
        p1_ind[i]=sub2ind(sub,conn_num_dims-1,conn_dims);
        //cout << "p1 ind is " << p1_ind[i] << endl;
    }

    for (mwSize i=0; i<p1_dims[1]; i++){
        mwSize sub[conn_num_dims-1];
        for (mwSize j=0; j<conn_num_dims-1; j++){
            sub[j]=p2_sub[i*(conn_num_dims-1)+j];
        }
        p2_ind[i]=sub2ind(sub,conn_num_dims-1,conn_dims);
            //cout << "p2 ind is " << p2_ind[i] << endl;
    }

    bool * connected = (bool *) mxMalloc(sizeof(bool)*p1_dims[1]);
    for (mwSize i=0; i<p1_dims[1]; i++){
        connected[i]=false;
    }
    //p1_ind=sub2ind(sub,conn_num_dims-1,conn_dims);
    //p2_ind=sub2ind(p2_sub,conn_num_dims-1,conn_dims);

    while (!pqueue.empty() && !stop){
        cur_edge=pqueue.top();
        //cout << cur_edge << endl;
        pqueue.pop();
        ind2sub(cur_edge,conn_num_dims,conn_dims,edge_array);
        mwSize v1, v2;
        mwSize v1_array[conn_num_dims-1], v2_array[conn_num_dims-1];
        for (mwSize i=0; i<conn_num_dims-1; i++){
            v1_array[i]=edge_array[i];
            v2_array[i]=edge_array[i];
        }
        for (mwSize i=0; i<nhood_dims[1]; i++){
            v2_array[i]+=nhood_data[nhood_dims[0]*i+edge_array[conn_num_dims-1]];
        }
        bool OOB=false;
        for (mwSize i=0; i<conn_num_dims-1; i++){
            if (v2_array[i]<0 || v2_array[i]>=conn_dims[i]){
                OOB=true;
            }
        }

        if (!OOB){
            v1=sub2ind(v1_array, conn_num_dims-1, conn_dims);
            v2=sub2ind(v2_array, conn_num_dims-1, conn_dims);
            mwSize set1=dsets.find_set(v1);
            mwSize set2=dsets.find_set(v2);
            if (set1!=set2){
                //mst_data[cur_edge]=1;
                dsets.link(set1, set2);
            }
        }

        mwSize count=0;
        for (mwSize i=0; i<p1_dims[1]; i++){        
            if ((dsets.find_set(p1_ind[i])==dsets.find_set(p2_ind[i])) && (!connected[i])){

                for (mwSize j=0; j<conn_num_dims; j++){
                    // switch from 0-base to 1-base
                    conn_edge_data[j+i*conn_num_dims]=edge_array[j]+1;
                    connected[i]=true;

                }
                
                count++;

            }
        }


        if (count==p1_dims[1]){
            stop=true;
        }
    }
    mxFree(connected);
}
