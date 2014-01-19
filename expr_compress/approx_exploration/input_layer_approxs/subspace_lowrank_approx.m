function [Wapprox, F, C, X, Y] = subspace_lowrank_approx(W, args)
% This approximation performs subspace clustering on the convolution
% kernels. Filters in the same cluster are approximated by a sum of rank
% one tensors. The reconstructed weight matrix, Wapprox, is returned along
% with the four rank one vectors (or dimensionality equal to respective
% input dimensions) and can be used to more efficiently compute the output
% of the convolution.
%
% W : dimensions (Fout, X, Y, Fin)
% args.num_colors : number of clusters (or "colors") to use
% args.terms_per_element : for each cluster of size m, we will use f*m rank
%                          one tensors to approximate the (m, X, Y, Fin)
%                          dimensional tensor associated with the cluster

    W = permute(W, [1, 4, 2, 3]);
    
    % Find clusters.
    lambda=0.001;
    WW=W(:, :);
    CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
    CKSym = BuildAdjacency(CMat, 0);
    [Grps] = SpectralClustering(CKSym, args.num_colors);
    order= Grps(:, 2); % Take results from second spectral clustering method.

    % Compress each cluster.
    Wapprox = zeros(size(W));
    for l = 1 : args.num_colors
        I = find(order == l);
        if ~isempty(I)
            chunk=W(I, :, :, :);
            [F{l}, C{l}, X{l}, Y{l}, cappr]=rankoneconv(chunk, args.terms_per_element * length(I));
            Wapprox(I, :, :, :)=cappr;
        end
    end
    
    % Pertume dimensions abck to original form
    Wapprox = permute(Wapprox, [1, 3, 4, 2]);
end
