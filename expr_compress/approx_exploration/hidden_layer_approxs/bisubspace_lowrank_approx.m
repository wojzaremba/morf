function [Wapprox, F, C, X, Y] = bisubspace_lowrank_approx(W, args)
% This approximation performs bi-clustering on input and output feature
% coordinates. After clustering, each kernel is then approximated by a sum
% of k rank one tensors. 
%
% W : dimensions (Fout, X, Y, Fin)
% args.iclust : Number of groups in input partition
% args.oclust : Number of groups in output partition
% args.k : Rank of tensor approximation (sum of k rank one tensors)

    % Find partition of input and output coordinates.
    X = W(:,:);
    idx_output = litekmeans(X', args.oclust);
    X = permute(W, [4 2 3 1]);
    X = X(:, :);
    idx_input = litekmeans(X', args.iclust);
    rast=1;
    
    % Now compress each cluster.
    Wapprox = zeros(size(W));
    for i=1:oclust(l)
        for j=1:iclust(l)
            Io=find(idx_output==i);
            Ii=find(idx_input==j);
            chunk = W(Io,:,:,Ii);
            
            %Compute a low-rank approximation of the kernel.
            [F{rast}, C{rast}, X{rast}, Y{rast}, cappr] = rankoneconv(chunk, args.k);
            rast = rast + 1;
            Wapprox(Io, :, :, Ii)=cappr;
        end
    end

end

