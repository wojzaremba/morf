function [Wapprox, F, C, X, Y, assignment, num_weights] = bisubspace_lowrank_approx_nosep(W, args)
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
    keyboard;
    idx_output = subspace_cluster(WW', args.oclust);
    WW = permute(W, [4 2 3 1]);
    WW = WW(:, :);
    idx_input = subspace_cluster(WW', args.iclust);
   
    rast=1;
        
    % Now compress each cluster.
    Wapprox = zeros(size(W));
    for i = 1 : args.oclust
        for j = 1 : args.iclust
            Io = find(idx_output == i);
            Ii = find(idx_input == j);
            chunk = W(Io, :, :, Ii);
            chunk = permute(chunk, [1 4 2 3]);
            chunk0 = reshape(chunk, size(chunk,1), size(chunk,2), size(chunk,3)*size(chunk,4));
            
            %Compute a low-rank approximation of the kernel.
            [F{rast}, C{rast}, X{rast}, Y{rast}, cappr] = rankoneconv3D(chunk0, args.k);
            Wapprox(Io, :, :, Ii)=permute(reshape(cappr, size(chunk)),[1 3 4 2]);
            assignment{rast}.Io = Io;
            assignment{rast}.Ii = Ii;
            rast = rast + 1;
        end
    end
    
    iclust_sz = size(W, 4) / args.iclust;
    oclust_sz = size(W, 1) / args.oclust;
    num_weights = (iclust_sz + oclust_sz + size(W, 2) + size(W, 2)) * args.iclust * args.oclust * args.k;

end

