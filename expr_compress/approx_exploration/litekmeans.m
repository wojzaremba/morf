function [outlabel,outm] = litekmeans(X, k)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).
% Modified by Joan Bruna to have constant cluster sizes

n = size(X,2);
last = 0;

minener = 1e+20;
outiters=16;
maxiters=200;

for j=1:outiters

aux=randperm(n);
m = X(:,aux(1:k));
%[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
[label] = constrained_assignment(X, m,n/k);

iters=0;


while any(label ~= last) & iters < maxiters
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    %[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
    [label,ener] = constrained_assignment(X, m,n/k);
    iters = iters +1 ;
end
[~,~,label] = unique(label);

if ener < minener
    outlabel = label;
    outm = m;
    minener = ener;
end

end


end



function [out,ener]=constrained_assignment(X, C, K)
%we assign samples to the nearest centers, but with the constraint that each center receives K samples
w=kernelizationbis(X',C');

[N,M]=size(w); %N number of samples, M number of centers

maxvalue = max(w(:))+1;

[ds,I]=sort(w,2,'ascend');
%[ds2,I2]=sort(w,1,'ascend');

out=I(:,1);
for m=1:M
    taille(m)=length(find(out==m));
end
[hmany,nextclust]=max(taille);

visited=zeros(1,M);


go=(hmany > K);
choices=ones(N,1);

while go
    %fprintf('%d %d \n', nextclust, hmany)
    aux=find(out==nextclust);

    for l=1:length(aux)
        slice(l) = ds(aux(l),choices(aux(l))+1)-ds(aux(l),choices(aux(l)));
    end
    [~,tempo]=sort(slice,'descend');
    clear slice;
    %slice=w(aux,nextclust);
    %[~,tempo]=sort(slice,'ascend');
    
    saved=aux(tempo(1:K));
    out(saved)=nextclust;

    visited(nextclust)=1;
    for k=K+1:length(tempo)
       i=2;
       while visited(I(aux(tempo(k)),i)) 
          i=i+1;
       end
       out(aux(tempo(k)))=I(aux(tempo(k)),i);
       choices(aux(tempo(k)))=i;
    end
    for m=1:M
        taille(m)=length(find(out==m));
    end
    [hmany,nextclust]=max(taille);
    go=(hmany > K);
end

ener=0;
for n=1:N
ener=ener+w(n,out(n));
end

end


%aux{m}=find(out==m);
%[~,order]=sort(taille,'descend');
%clear aux;

        %w(saved,:)=maxvalue;
        %w(:,m)=maxvalue;
        %w(saved,m)=0;
        %[ds,I]=sort(w,2,'ascend');
        %out=I(:,1);

%%for m=1:M
%    aux=find(out==nextclust);
%    if length(aux)>K
%        slice=w(aux,order(m));
%        [~,tempo]=sort(slice,'ascend');
%        saved=aux(tempo(1:K));
%        out(saved)=order(m);
%        %visited(order(m))=1;
%        I(find(I==order(m)))=0;
%        for k=K+1:length(tempo)
%            i=2;
%            while I(aux(tempo(k)),i)==0
%                i=i+1;
%            end
%            out(aux(tempo(k)))=I(aux(tempo(k)),i);
%        end
%        %w(saved,:)=maxvalue;
%        %w(:,m)=maxvalue;
%        %w(saved,m)=0;
%        %[ds,I]=sort(w,2,'ascend');
%        %out=I(:,1);
%    end
%end
%
%
