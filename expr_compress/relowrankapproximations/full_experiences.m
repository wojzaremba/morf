

clear all;
addpath(genpath('.'));
%load('/Users/joan/Documents/matlab/morf_tmp/morf/compression/all_imagenet_weights.mat');
%load('/Users/joan/Documents/matlab/morf_tmp/morf/compression/imagenet_weights_v2.mat');

%the sizes are (output H V input)
load /misc/vlgscratch2/LecunGroup/bruna/all_imagenet_weights.mat;

L=2;
oclust=[32 32];
iclust=[1 3]; 
P=[20 40];

for l=1:min(L,size(layer,2))

    %compute the spectrum of the linear operator
    spectrum{l} = convtensor_frame_bounds(layer{l});

    %compute a low-rank approximation of the kernel
    
    %bi-clustering on input and output feature coordinates
    X = layer{l}(:,:);
    idx_output = litekmeans(X',oclust(l));
    X = permute(layer{l},[4 2 3 1]);
    X = X(:,:);
    idx_input = litekmeans(X',iclust(l));
    rast=1;
    for i=1:oclust(l)
        for j=1:iclust(l)
            Io=find(idx_output==i);
            Ii=find(idx_input==j);
            chunk = layer{l}(Io,:,:,Ii);
            %size(chunk)
            [~,~,~,~,aerr{rast}]=rankoneconvtemp(chunk,P(l));rast=rast+1;
        end
    end

end

