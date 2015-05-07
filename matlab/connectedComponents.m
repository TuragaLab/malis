function [comp,cmpSz] = connectedComponents(conn,nhood2,nhood1,sizeThreshold)
% CONNECTEDCOMPONENTS	computes the connected components of a binary graph
% [comp,cmpSz] = connectedComponents(conn,nhood,sizeThreshold)
% conn			a 4d 'conn' style connectivity graph
% nhood			the neighborhood associated with the 'conn' graph giving the interpretation of the edges (default is the nearest neighbor 6-connectivity corresponding to 3 edges per voxel)
% sizeThreshold	objects that have as many voxels or fewer than this threshold are treated as 'dust' and removed from the connected components output
%
% comp			the output connected components labeling (sorted in descending order of size)
% cmpSz			the sizes of each component

%% developed and maintained by Srinivas C. Turaga <sturaga@mit.edu>
%% do not distribute without permission.

sz = size(conn);

% 6-connectivity is the default
if ~exist('nhood2','var') || isempty(nhood2),
	nhood2 = -eye(ndims(conn)-1);
	nhood1 = zeros(size(nhood2));
end

if ~exist('nhood1','var') || isempty(nhood1),
	nhood1 = zeros(size(nhood2));
end

if size(nhood1) ~= size(nhood2),
	error('nhood1 and nhood2 do not have the same size!')
end

% size threshold for pruning objects
if ~exist('sizeThreshold','var') || isempty(sizeThreshold),
	sizeThreshold = 1;
end

if size(conn,ndims(conn)) ~= size(nhood1,1),
	error('number of edges don''t match')
end

% precondition input
conn = logical(conn~=0);
nhood1 = double(nhood1);
nhood2 = double(nhood2);

% run mex code
if ~any(nhood1(:)),
	[comp,cmpSz] = connectedComponentsmex(conn,nhood2);
else,
	[comp,cmpSz] = connectedComponentsmex(conn,nhood1,nhood2);
end

% remove "dust" (objects of size 1)
% and re-order in reverse order of size
[cmpSz,reverse_renum] = sort(cmpSz,'descend');
renum = zeros(1,length(reverse_renum),'uint32');
reverse_renum = uint32(reverse_renum);
dust = cmpSz<=sizeThreshold;
reverse_renum(dust) = [];
cmpSz(dust) = [];
numObj = length(cmpSz);

renum(reverse_renum) = 1:numObj;
comp = renum(comp);
comp = reshape(comp,sz(1:end-1));
