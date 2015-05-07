function [mmEdge,mmWt] = maximinEdge(conn,nhood,pt1,pt2)
%% MAXIMINEDGE	Compute the maximin edge between two points pt1 and pt2

conn = single(conn);
nhood = double(nhood);
% pt1 = pt1(:);
% pt2 = pt2(:);

mmEdge = maximinmex(conn,nhood,pt1,pt2);
mmWt = [];
%mmEdgeSub = num2cell(mmEdge);
%mmWt = conn(sub2ind(size(conn),mmEdgeSub{:}));
