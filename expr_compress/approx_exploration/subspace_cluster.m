function [Grps] = subspace_cluster(X, K)

lambda=0.001;
CMat = SparseCoefRecovery(X,0,'Lasso',lambda);
CKSym = BuildAdjacency(CMat,0);

N = size(CKSym,1);
%MAXiter = 1000; % Maximum iteration for KMeans Algorithm
%REPlic = 100; % Replication for KMeans Algorithm
% Method 2: Random Walk Method
DKN=( diag( sum(CKSym) ) )^(-1);
LapKN = speye(N) - DKN * CKSym;
[uKN,sKN,vKN] = svd(LapKN);
f = size(vKN,2);
kerKN = vKN(:,f-K+1:f);
svalKN = diag(sKN);
Grps = litekmeans(kerKN',K);



