
clear all;

clearvars -except root_path plan;
addpath(genpath('.'))
load('imagenet_weights.mat');

%simple re-parametrization of first layer with monochromatic filters
W = layer1;
for f=1:size(W,1)
    [u,s,v]=svd(squeeze(W(f,:,:)),0);
    C(f,:)=u(:,1);
    S(f,:)=v(:,1);
    dec(f,:)=diag(s);
    chunk = u(:,1)*s(1,1)*v(:,1)';
    approx0(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
end

L=4;%number of 'colors'

MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 10; % Replication for KMeans Algorithm
[assignment,colors] = litekmeans(C',L);
[ordsort,perm]=sort(assignment);

colors = colors';
Wapprox = zeros(size(W));
for f=1:size(W,1)
    chunk = (colors(assignment(f),:)')*dec(f,1)*(S(f,:));
    Wapprox(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
end


Wmono = reshape(bsxfun(@times, S, dec(:, 1)),size(W,1),size(W,3),size(W,4));
%Wmono = Wmono(perm, :, :);
collage_imnet(W); 
collage_imnet(Wapprox);
                       
Wconstruct = zeros(96, 11, 11, 3);
for i = 1 : size(W, 1)
    for c = 1 : 3
        Wconstruct(i, :, :, c) =  Wmono(i, :, :) * colors(assignment(i), c);
    end
end

Wapprox = permute(Wapprox, [1, 3, 4, 2]);

 norm(squeeze(Wmono(1, :, :)) * colors(assignment(1), 1) + ...
  squeeze(Wmono(1, :, :)) * colors(assignment(1), 2) + ...
  squeeze(Wmono(1, :, :)) * colors(assignment(1), 3) - ...
(squeeze(Wapprox(1, :, :, 1)) + squeeze(Wapprox(1, :, :, 2)) + squeeze(Wapprox(1, :, :, 3))));


collage_imnet(permute(Wconstruct, [1, 4, 2, 3]));