
clear all;

load('imagenet_weights.mat');


%simple re-parametrization of first layer with monochromatic filters

for f=1:size(layer1,1)
    [u,s,v]=svd(squeeze(layer1(f,:,:)),0);
    C(f,:)=u(:,1);
    S(f,:)=v(:,1);
    dec(f,:)=diag(s);
    chunk = u(:,1)*s(1,1)*v(:,1)';
    approx0(f,:,:,:)=reshape(chunk,1,size(layer1,2),size(layer1,3),size(layer1,4));
end

L=4;%number of 'colors'
if 0
    lambda=0.001;
    CMat = SparseCoefRecovery(C',0,'Lasso',lambda);
    CKSym = BuildAdjacency(CMat,0);
    [Grps] = SpectralClustering(CKSym,L);
    order= Grps(:,2);
else
    MAXiter = 1000; % Maximum iteration for KMeans Algorithm
    REPlic = 10; % Replication for KMeans Algorithm
    [order,colors] = kmeans(C,L,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
end
[ordsort,popo]=sort(order);

for f=1:size(layer1,1)
    chunk = (colors(order(f),:)')*dec(f,1)*(S(f,:));
    approx1(f,:,:,:)=reshape(chunk,1,size(layer1,2),size(layer1,3),size(layer1,4));
end

codeC=order;
C=colors;
S=reshape(S,size(layer1,1),size(layer1,3),size(layer1,4));

collage_imnet(layer1);
collage_imnet(approx1);


save('layer1color.mat','codeC','C','S');




