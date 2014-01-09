
close all;
%layer 1 by default. 
aux = double(plan.layer{2}.params.W);

[L, K, N, M]=size(aux);
sepcut=2;

horizontals=[];
verticals=[];

for l=1:L
chunk = squeeze(aux(l,:,:,:));
chunk = chunk(:,:);
%mu(l,:)= mean(chunk,2)';
%chunk = chunk-repmat(mu(l,:)',[1 size(chunk,2)]);
[u,s,v]=svd(chunk',0);
coldecay(l,:)=diag(s);
chunkbis=v(:,1)*s(1,1)*u(:,1)';
auxbis(l,:,:,:)=reshape(chunkbis,K,N,M);
if v(1,1)>0
point(:,l)=v(:,1);
else
point(:,l)=-v(:,1);
end
slice=reshape(u(:,1),11,11);
[ut,st,vt]=svd(slice,0);
tempo = ut(:,1:sepcut)*st(1:sepcut,1:sepcut)*vt(:,1:sepcut)';
chunk=v(:,1)*s(1,1)*reshape(tempo,1,N*M);
auxbis2(l,:,:,:)=reshape(chunk,K,N,M);
horizontals=[horizontals ut(:,1:sepcut)];
verticals=[verticals vt(:,1:sepcut)];
end


%oldparams=aux;
%plan.layer{2}.params.W = auxbis2;
%[va, vb]=kmeans(verticals',50);

kk=L/2;
[U,O]=subspace_subsample(double(aux(:,:)'),kk);


%OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
%lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
%KK = 0;
%data=double(aux(:,:)');
%CMat = SparseCoefRecovery(data,0,OptM,lambda);
%CKSym = BuildAdjacency(CMat,KK);
%[Grps , SingVals, LapKernel] = SpectralClustering(CKSym,kk);
%




K=4;
[a,b]=kmeans(point',K);

[~,visit]=sort(a);


S=size(aux);

N=16;
M=floor(S(1)/N);

marg=2;

collage = zeros( (S(3)+marg)*N, (S(4)+marg)*M, 3);
collage2 = zeros( (S(3)+marg)*N, (S(4)+marg)*M, 3);
collage3 = zeros( (S(3)+marg)*N, (S(4)+marg)*M, 3);

for rast=1:S(1)
rn = mod(rast-1,N)+1;
rm = floor((rast-1)/N)+1;
collage( (S(3)+marg)*(rn-1)+1:(S(3)+marg)*(rn-1)+S(3), (S(4)+marg)*(rm-1)+1:(S(4)+marg)*(rm-1)+S(4),:)=colorescale(permute(squeeze(aux(visit(rast),:,:,:)),[2 3 1]));
%collage2( (S(3)+marg)*(rn-1)+1:(S(3)+marg)*(rn-1)+S(3), (S(4)+marg)*(rm-1)+1:(S(4)+marg)*(rm-1)+S(4),:)=colorescale(permute(squeeze(auxbis(visit(rast),:,:,:)),[2 3 1]));
collage3( (S(3)+marg)*(rn-1)+1:(S(3)+marg)*(rn-1)+S(3), (S(4)+marg)*(rm-1)+1:(S(4)+marg)*(rm-1)+S(4),:)=colorescale(permute(squeeze(auxbis2(visit(rast),:,:,:)),[2 3 1]));

end
figure(1)
imagesc(collage)
%figure(2)
%imagesc(collage2)
figure(3)
imagesc(collage3)

figure(4)
plot(a(visit))

