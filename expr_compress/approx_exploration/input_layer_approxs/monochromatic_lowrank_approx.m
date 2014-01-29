function [Wapprox, colors, filters, filter_coeff, perm] = monochromatic_lowrank_approx(W, args)
    % This approximation combines two different approximation techniques to
    % approximate the first layer filters of a conv ent. First, the color
    % components are replaced by a rank-1 approximation. The weights are now
    % monochromatic. These monochromatic weights are them further approximated
    % by clustering the filters using subspace clutering and then approximating
    % the tensors corresponding to each cluster with a low rank tensor. The 
    % reconstructed  weight matrix, Wapprox, is returned along with the
    % matrices that can be used to more efficiently compute the output of the 
    % convolution.
    %
    % args.num_clusterss : number of clusters to use
    % args.rank : rank to use for low-rank approximation
    %
    % colors : (96, 3)
    % filters : (K*K, rank, num_clusters), where K is size of filter
    % filter_coeff : (cluster_size, rank, num_clusters)
    % perm : (96, 1)


    W = permute(W, [1, 4, 2, 3]);
    approx0  =zeros(size(W));
    for f = 1 : size(W,1)
        [u,s,v]=svd(squeeze(W(f,:,:)),0);
        C(f, :) = u(:, 1);
        S(f, :) = s(1, 1) * v(:, 1); %v(:,1);
    end

    colors = C;
    % Now cluster the mono weights
    Wmono = reshape(S,size(W,1),size(W,3),size(W,4));
    assignment = litekmeans(Wmono(:, :)', args.num_clusters);
    % lambda=0.001;
    % CMat = SparseCoefRecovery(double(Wmono(:, :)'), 0, 'Lasso', lambda);
    % CKSym = BuildAdjacency(CMat, 0);
    % [Grps] = SpectralClusteringEven(CKSym, args.num_clusters);
    % assignment= Grps(:, 2);
    [~, perm] = sort(assignment);

    Wmonoapprox = zeros(size(Wmono));
    filters = zeros(size(W, 3) * size(W, 4), args.rank, args.num_clusters);
    filter_coeff = zeros(size(W, 1) / args.num_clusters, args.rank, args.num_clusters);
    for l = 1 : args.num_clusters
        I = find(assignment == l);
        if ~isempty(I)
            chunk=Wmono(I, :, :);
            [u, s, v] = svd(chunk(:, :));
            filt = v(:, 1:args.rank);
            coeff = u(:, 1:args.rank) *s(1:args.rank, 1:args.rank);
            cappr = coeff * filt';
            Wmonoapprox(I, :)=cappr;
            filters(:, :, l) = filt;
            filter_coeff(:, :, l) = coeff;
        end
    end

    % filters : (K*K, rank, num_clusters), where K is size of filter
    % filter_coeff : (cluster_size, rank, num_clusters)

    Wapprox = zeros(size(W));
    for f=1:size(W,1)
        chunk = colors(f,:)' * Wmonoapprox(f,:);
        Wapprox(f,:,:,:)=reshape(chunk,1,size(W,2),size(W,3),size(W,4));
    end 

    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
end

