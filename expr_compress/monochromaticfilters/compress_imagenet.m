
load('/Users/joan/Documents/matlab/morf_tmp/morf/compression/imagenet_weights.mat');


%1 . first layer

L=48;%number of clusters
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

if 0

    %2 . second layer
    %L=24, f=60 yields 107 / 128
    %L=48, f=30 yields 108 / 128 OK
    %L=48, f=24 yields 108 / 128 OK

    clear layer2_appr;
    laybis=permute(layer2,[2 1 3 4]);
    L=48;%number of clusters
    f =24 ; %number of terms per element
    X=laybis(:,:);
    if 0
        lambda=0.001;
        CMat = SparseCoefRecovery(X',0,'Lasso',lambda);
        CKSym = BuildAdjacency(CMat,0);
        [Grps] = SpectralClustering(CKSym,L);
        order2= Grps(:,2);
    else
        MAXiter = 1000; % Maximum iteration for KMeans Algorithm
        REPlic = 10; % Replication for KMeans Algorithm
        order2 = kmeans(X,L,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
    end

    [ordsort2,popo2]=sort(order2);
    layer2bis=laybis(popo2,:,:,:);
    S=size(laybis,1);
    %compress each cluster
    for l=1:S/L:S
        chunk=layer2bis(l:l+S/L-1,:,:,:);
        fprintf('%d elements with %d rank1-tensors \n', S/L, f)
        [C2{l},H2{l},V2{l},F2{l},cappr]=rankoneconv(chunk,f);
        layer2_appr(l:l+S/L-1,:,:,:)=cappr;
    end
    popo2inv=inversperm(popo2);
    layer2_appr = layer2_appr(popo2inv,:,:,:);
    layer2_appr = permute(layer2_appr,[2 1 3 4]);
    %I=find(order2==l);
    %if ~isempty(I)
    %chunk=layer2(I,:,:,:);
    %[C2{l},H2{l},V2{l},F2{l},cappr]=rankoneconv(chunk,f*length(I));
    %layer2_appr(I,:,:,:)=cappr;
    %end
    %end


    save('/Users/joan/Documents/matlab/morf/trained/imagenet_appr_weights.mat','layer1_appr','layer2_appr','layer1','layer2',...
    'C','H','V','F','C2','H2','V2','F2');

end


