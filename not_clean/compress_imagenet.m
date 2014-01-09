
load('/Users/joan/Documents/matlab/morf/trained/imagenet_weights.mat');


%1 . first layer

L=24;%number of clusters
f = 2; %number of terms per element

lambda=0.001;
X=layer1(:,:);
CMat = SparseCoefRecovery(X',0,'Lasso',lambda);
CKSym = BuildAdjacency(CMat,0);
[Grps] = SpectralClustering(CKSym,L);
order= Grps(:,2);

[ordsort,popo]=sort(order);
%layer1=layer1(popo,:,:,:);

%compress each cluster
for l=1:L
I=find(order==l);
if ~isempty(I)
chunk=layer1(I,:,:,:);
[C{l},H{l},V{l},F{l},cappr]=rankoneconv(chunk,f*length(I));
layer1_appr(I,:,:,:)=cappr;
end
end


%2 . second layer

L=32;%number of clusters
f =4 ; %number of terms per element

lambda=0.001;
X=layer2(:,:);
CMat = SparseCoefRecovery(X',0,'Lasso',lambda);
CKSym = BuildAdjacency(CMat,0);
[Grps] = SpectralClustering(CKSym,L);
order2= Grps(:,2);

[ordsort2,popo2]=sort(order2);
%compress each cluster
for l=1:L
I=find(order2==l);
if ~isempty(I)
chunk=layer2(I,:,:,:);
[C2{l},H2{l},V2{l},F2{l},cappr]=rankoneconv(chunk,f*length(I));
layer2_appr(I,:,:,:)=cappr;
end
end


save('/Users/joan/Documents/matlab/morf/trained/imagenet_appr_weights.mat','layer1_appr','layer2_appr');



