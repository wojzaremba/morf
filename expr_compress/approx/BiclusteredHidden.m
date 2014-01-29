classdef BiclusteredHidden < Approximation
    properties  
    end
    
    methods(Static)
        function Wapprox = ReconstructW(colors, dec, S, assignment, dims)
            assert(0);
        end
    end
    
    methods
        function obj = BiclusteredHidden(general_vars, approx_vars, cuda_vars)
            obj@Approximation(general_vars, approx_vars, cuda_vars);
            obj.name = 'biclustered_hidden';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            assert(0);
        end        
        
        function [Wapprox, ret] = Approx(obj, params)   
            % params.iclust : number of input clusters
            % params.oclust : number of output clusters
            % params.rank : rank of approx for each cluster
            % params.cluster_type : kmeans or subspace
            global plan;
            global root_path;
            
            % Get original W, b
            if (plan.layer{5}.on_gpu)
                W = C_(CopyFromGPU, plan.layer{2}.gpu.vars.W);
                W = reshape(W, [96, 11, 11, 3]);
            else
                W = double(plan.layer{5}.cpu.vars.W);
            end
            
            iclust_sz = size(W, 4) / params.iclust;
            oclust_sz = size(W, 1) / params.oclust;

            % Find partition of input and output coordinates.
            if (strcmp(params.cluster_type, 'kmeans'))
                MAXiter = 1000; % Maximum iteration for KMeans Algorithm
                REPlic = 100;
                WW = W(:,:);
                idx_output = litekmeans(WW', params.oclust);
                WW = permute(W, [4 2 3 1]);
                WW = WW(:, :);
                idx_input = litekmeans(WW', params.iclust);
            elseif (strcmp(params.cluster_type, 'subspace'))
                lambda=0.001;
                WW=W(:, :);
                CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
                CKSym = BuildAdjacency(CMat, 0);
                [Grps] = SpectralClustering(CKSym, params.oclust);
                idx_output= Grps(:, 2); 

                WW = permute(W, [4 2 3 1]);
                WW = WW(:, :);
                CMat = SparseCoefRecovery(WW', 0, 'Lasso', lambda);
                CKSym = BuildAdjacency(CMat, 0);
                [Grps] = SpectralClustering(CKSym, params.iclust);
                idx_input= Grps(:, 2); 
            else
                assert(0);
            end

            [~, perm_in] = sort(idx_input);
            [~, perm_out] = sort(idx_output);

            rast=1;

            % Now compress each cluster.
            Wapprox = zeros(size(W));
            F = zeros([params.oclust, oclust_sz, params.rank, params.iclust]);
            C = zeros([params.oclust, iclust_sz, params.rank, params.iclust]);
            XY = zeros([params.oclust, size(W, 2)^2,params.rank, params.iclust]);
            for i = 1 : params.oclust
                for j = 1 : params.iclust
                    Io = find(idx_output == i);
                    Ii = find(idx_input == j);
                    chunk = W(Io, :, :, Ii);

                    %Compute a low-rank approximation of the kernel.
                    [f, x, y, c, cappr] = rankoneconv(chunk, params.rank);
                    F(i, :, :, j) = f;
                    C(i, :, :, j) = c;
                    xy = zeros(size(W, 2) * size(W, 3), params.rank);
                    for ii = 1 : params.rank
                       xy(:, ii) = reshape(x(:, ii) * y(:, ii)', size(W, 2) * size(W, 3), 1); % Not right
                    end
                    XY(i, :, :, j) = xy;
                    Wapprox(Io, :, :, Ii)=cappr;
                    rast = rast + 1;
                end
            end
            
            ret.Wapprox = Wapprox;
            ret.layer = 'BiclusteredConv';
            ret.layer_nr = 5;
            ret.vars.F = F;
            ret.vars.C = C;
            ret.vars.XY = XY;
            ret.vars.perm_in = perm_in; 
            ret.vars.perm_out = perm_out;
            ret.json = struct('iclust', params.iclust, 'oclust', params.oclust, 'rank', params.rank);
            ret.on_gpu = Val(params, 'on_gpu', obj.on_gpu);
            
        end       

    end
    
end
