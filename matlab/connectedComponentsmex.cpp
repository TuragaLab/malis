/* Connected components
 * developed and maintained by Srinivas C. Turaga <sturaga@mit.edu>
 * do not distribute without permission.
 */

#include "mex.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stack>
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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
	// input mapping
	const mxArray * conn = prhs[0];
	const mwSize conn_num_dims = mxGetNumberOfDimensions(conn);
	const mwSize * conn_dims = mxGetDimensions(conn);
	const mwSize conn_num_elements = mxGetNumberOfElements(conn);
	const mxLogical * conn_data = mxGetLogicals(conn);
	const mxArray * nhood1 = prhs[1];
	const mwSize nhood1_num_dims = mxGetNumberOfDimensions(nhood1);
	const mwSize * nhood1_dims = mxGetDimensions(nhood1);
	const double * nhood1_data = mxGetPr(nhood1);
	if (nhood1_num_dims != 2) {
		mexErrMsgTxt("wrong size for nhood1");
	}
	if ((nhood1_dims[1] != (conn_num_dims-1))
		|| (nhood1_dims[0] != conn_dims[conn_num_dims-1])){
		mexErrMsgTxt("nhood1 and conn dimensions don't match");
	}
	const mxArray * nhood2 = prhs[1];
	const mwSize nhood2_num_dims = mxGetNumberOfDimensions(nhood2);
	const mwSize * nhood2_dims = mxGetDimensions(nhood2);
	const double * nhood2_data = mxGetPr(nhood2);
	// output mapping
	mxArray * label;
	mwSize label_num_dims = conn_num_dims-1;
	mwSize label_dims[conn_num_dims - 1];
	for (mwSize i=0; i<(conn_num_dims-1); i++){
		label_dims[i] = conn_dims[i];
	}
	plhs[0] = mxCreateNumericArray(label_num_dims,label_dims,mxUINT32_CLASS,mxREAL);
	if (plhs[0] == NULL) {
		mexErrMsgTxt("Unable to create output array");
		return;
	}

	label=plhs[0];
	uint32_t * label_data = (uint32_t *) mxGetData(label);
	mwSize label_num_elements = mxGetNumberOfElements(label);

	// initialize colors (a node is either discovered or not), maybe change to integer?
	//bool discovered[label_num_elements];
	bool * discovered = (bool *) mxMalloc(label_num_elements*sizeof(bool));
	for (mwSize i=0; i<label_num_elements; i++){
		discovered[i]=false;
	}

	std::stack<mwSize> S;
	std::vector<mwSize> component_sizes;
	for (mwSize ind=0; ind<label_num_elements; ind++){
		if (discovered[ind]==false){
			S.push(ind);
			component_sizes.push_back(1);
			label_data[ind]=component_sizes.size();
			discovered[ind]=true;
			mwSize current;
			while (!S.empty()){
				current=S.top();
				mwSize cur_pos[label_num_dims];
				ind2sub(current, label_num_dims, label_dims, cur_pos);
				S.pop();
				mwSize nbor[conn_num_dims];
				mwSize nbor_ind;
				mwSize new_pos[label_num_dims];
				mwSize new_ind;

				for (int i=0; i<label_num_dims; i++){
					nbor[i]=cur_pos[i];
				}
				for (int i=0; i<nhood1_dims[0]; i++){
					nbor[conn_num_dims-1]=i;
					nbor_ind=sub2ind(nbor,conn_num_dims,conn_dims);
					if (conn_data[nbor_ind]){
						bool OOB=false;
						for (mwSize j=0; j<label_num_dims; j++){
							
							mwSize check=cur_pos[j]+(mwSize) nhood1_data[i+j*nhood1_dims[0]];
							if (check<0 || check>=label_dims[j]){
								OOB=true;
							}
							new_pos[j]=check;
						}
						if (!OOB){
							new_ind=sub2ind(new_pos,label_num_dims,label_dims);
							if (!discovered[new_ind]){
								S.push(new_ind);
								label_data[new_ind]=component_sizes.size();
								discovered[new_ind]=true;
								component_sizes.back()+=1;
							}
						}
					}
				}

				for (mwSize i=0; i<nhood1_dims[0]; i++){
					bool OOB=false;
					for (mwSize j=0; j<label_num_dims; j++){
						mwSize check=cur_pos[j]- (mwSize) nhood1_data[i+j*nhood1_dims[0]];
						if (check<0 || check>=label_dims[j]){
							OOB=true;
						}
						new_pos[j]=check;
					}
					if (!OOB){
						for (mwSize j=0; j<label_num_dims; j++){
							nbor[j]=new_pos[j];
						}
						nbor[conn_num_dims-1]=i;
						nbor_ind=sub2ind(nbor, conn_num_dims, conn_dims);
						if (conn_data[nbor_ind]){
							new_ind=sub2ind(new_pos, label_num_dims, label_dims);
							if (!discovered[new_ind]){
								S.push(new_ind);
								label_data[new_ind]=component_sizes.size();
								discovered[new_ind]=true;
								component_sizes.back()+=1;
							}
						}
					}
				}
			}
		}
	}
	mwSize size_dims[2];
	size_dims[0]=component_sizes.size();
	size_dims[1]=1;
	plhs[1]=mxCreateNumericArray(2, size_dims, mxUINT32_CLASS, mxREAL);
	uint32_t * psizes = (uint32_t *) mxGetData(plhs[1]);
	for (mwSize i=0; i<(mwSize)component_sizes.size(); i++){
		psizes[i]=(uint32_t)component_sizes[i];
	}
	mxFree(discovered);
}
