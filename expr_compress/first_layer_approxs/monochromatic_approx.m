function [Wapprox, Wmono, colors, perm] = monochromatic_approx(W, args)
% This approximation clusters the first left singular vectors of each of
% the convolution kernels associated with each output feature. Filters in 
% the same cluster share the same inner color component. The reconstructed 
% weight matrix, Wapprox, is returned along with the the color
% transformation matrix, the monochromatic weights and the permutation of
% the weights. These matrices can be used to more efficiently compute the 
% output of the convolution.
%
% args.even : 1 if clusters should be constrained to be equal sizes, 0 
%             otherwise
% args.num_colors : number of clusters (or "colors") to use

    W = permute(W, [1, 4, 2, 3]);
    % Simple re-parametrization of first layer with monochromatic filters
    for f = 1 : size(W,1)
        [u,s,v]=svd(squeeze(W(f,:,:)),0);
        C(f,:)=u(:,1);
        S(f,:)=v(:,1);
        dec(f,:)=diag(s);
        chunk = u(:,1)*s(1,1)*v(:,1)';
        approx0(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end
    
    if args.even
        [assignment,colors] = litekmeans(C', args.num_colors);
        colors = colors';
    else
        MAXiter = 1000; % Maximum iteration for KMeans Algorithm
        REPlic = 100; % Replication for KMeans Algorithm
%        [assignment,colors] = kmeans(C, args.num_colors, 'start', args.start, 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
         [assignment,colors] = kmeans(C, args.num_colors, 'start', 'sample', 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
    end
    
    Wapprox = zeros(size(W));
    for f=1:size(W,1)
        chunk = (colors(assignment(f),:)')*dec(f,1)*(S(f,:));
        Wapprox(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end 
    
    Wmono = reshape(bsxfun(@times, S, dec(:, 1)),size(W,1),size(W,3),size(W,4));
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
    
    assert(norm(squeeze(Wmono(1, :, :)) * colors(assignment(1), 1) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 2) + ...
                        squeeze(Wmono(1, :, :)) * colors(assignment(1), 3) - ...
                        (squeeze(Wapprox(1, :, :, 1)) + squeeze(Wapprox(1, :, :, 2)) + squeeze(Wapprox(1, :, :, 3)))) < 1e-10);
                    
    
    [~, perm] = sort(assignment);
    colors = colors';
end

