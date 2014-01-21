function [Wapprox, F, C, X, Y, assignment] = bisubspace_lowrank_approx(W, args)
% This approximation performs bi-clustering on input and output feature
% coordinates. After clustering, each kernel is then approximated by a sum
% of k rank one tensors. 
%
% W : dimensions (Fout, X, Y, Fin)
% args.iclust : Number of groups in input partition
% args.oclust : Number of groups in output partition
% args.k : Rank of tensor approximation (sum of k rank one tensors)

    % Find partition of input and output coordinates.
    WW = W(:,:);
    idx_output = litekmeans(WW', args.oclust);
    WW = permute(W, [4 2 3 1]);
    WW = WW(:, :);
    idx_input = litekmeans(WW', args.iclust);

%     lambda=0.001;
%     WW=W(:, :);
%     CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
%     CKSym = BuildAdjacency(CMat, 0);
%     [Grps] = SpectralClustering(CKSym, args.oclust);
%     idx_output= Grps(:, 2); 
%     
%     WW = permute(W, [4 2 3 1]);
%     WW = WW(:, :);
%     CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
%     CKSym = BuildAdjacency(CMat, 0);
%     [Grps] = SpectralClustering(CKSym, args.iclust);
%     idx_input= Grps(:, 2); 
   
    rast=1;
        
    % Now compress each cluster.
    Wapprox = zeros(size(W));
    for i = 1 : args.oclust
        for j = 1 : args.iclust
            Io = find(idx_output == i);
            Ii = find(idx_input == j);
            chunk = W(Io, :, :, Ii);
            
            %Compute a low-rank approximation of the kernel.
            [F{rast}, C{rast}, X{rast}, Y{rast}, cappr] = rankoneconv(chunk, args.k);
            Wapprox(Io, :, :, Ii)=cappr;
            assignment{rast}.Io = Io;
            assignment{rast}.Ii = Ii;
            rast = rast + 1;
        end
    end

end

