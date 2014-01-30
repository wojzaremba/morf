function [Wapprox, F, C, XY, perm_in, perm_out, num_weights] = bisubspace_lowrank_approx(W, args)
% This approximation performs bi-clustering on input and output feature
% coordinates. After clustering, each kernel is then approximated by a sum
% of k rank one tensors. 
%
% W : dimensions (Fout, X, Y, Fin)
% args.iclust : Number of groups in input partition
% args.oclust : Number of groups in output partition
% args.k : Rank of tensor approximation (sum of k rank one tensors)

    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;
    num_weights = (iclust_sz + oclust_sz + size(W, 2) * size(W, 2)) * args.iclust * args.oclust * args.k;

    % Find partition of input and output coordinates.
    if (strcmp(args.cluster_type, 'subspace'))
        lambda=0.001;
        WW=W(:, :);
        CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
        CKSym = BuildAdjacency(CMat, 0);
        [Grps] = SpectralClustering(CKSym, args.oclust);
        idx_output= Grps(:, 2); 
    
        WW = permute(W, [4 2 3 1]);
        WW = WW(:, :);
        CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
        CKSym = BuildAdjacency(CMat, 0);
        [Grps] = SpectralClustering(CKSym, args.iclust);
        idx_input= Grps(:, 2); 
    elseif (strcmp(args.cluster_type, 'kmeans'))
        MAXiter = 1000; % Maximum iteration for KMeans Algorithm
        REPlic = 100;
        WW = W(:,:);
        idx_output = litekmeans(WW', args.oclust);
        WW = permute(W, [4 2 3 1]);
        WW = WW(:, :);
        idx_input = litekmeans(WW', args.iclust);
    else
        assert(0);
    end
        
    [~, perm_in] = sort(idx_input);
    [~, perm_out] = sort(idx_output);
    
    rast=1;
        
    % Now compress each cluster.
    Wapprox = zeros(size(W));
    F = zeros([oclust_sz, args.k, args.iclust, args.oclust]);
    C = zeros([iclust_sz, args.k, args.iclust, args.oclust]);
    XY = zeros([size(W, 2)^2,args.k, args.iclust, args.oclust]);
    for i = 1 : args.oclust
        for j = 1 : args.iclust
            Io = find(idx_output == i);
            Ii = find(idx_input == j);
            chunk = W(Io, :, :, Ii);
            
            %Compute a low-rank approximation of the kernel.
            [f, x, y, c, cappr] = rankoneconv(chunk, args.k);
            F(:, :, j, i) = f;
            C(:, :, j, i) = c;
            xy = zeros(size(W, 2) * size(W, 3), args.k);
            for ii = 1 : args.k
               xy(:, ii) = reshape(x(:, ii) * y(:, ii)', size(W, 2) * size(W, 3), 1); % Not right
            end
            XY(:, :, j, i) = xy;
            Wapprox(Io, :, :, Ii)=cappr;
            rast = rast + 1;
        end
    end
end

