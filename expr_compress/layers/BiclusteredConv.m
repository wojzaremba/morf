classdef BiclusteredConv < LayerApprox
    properties
        iclust
        oclust
        rank
    end
    
    methods
        function obj = BiclusteredConv(json)
            obj@LayerApprox(FillDefault(json));
            obj.iclust = json.iclust;
            obj.oclust = json.oclust;
            obj.rank = json.rank;
            obj.Finalize();
        end                      
       
        function FP_(obj)
            assert(0);
        end
        
        function FP(obj)
            prev_dim = obj.prev_dim();
            v = obj.cpu.vars;
            X = v.X;  
            bs = size(X, 1);
            
            iclust_sz = prev_dim(3) / obj.iclust;
            oclust_sz = obj.dims(3) / obj.oclust;
            final_res = zeros(size(X, 1) * prod(obj.dims(1:2)), obj.dims(3));
            for oc = 1 : obj.oclust
                idx_out = v.perm_out( (oc - 1) * oclust_sz + 1 : oc * oclust_sz );
                res = zeros(size(X, 1) * prod(obj.dims(1:2)), oclust_sz);
                for ic = 1 : obj.iclust
                    idx_in = v.perm_in( (ic - 1) * iclust_sz + 1: ic * iclust_sz );
                    X_ = zeros(size(X, 1), prev_dim(1) + obj.padding(1) * 2 + obj.patch(1), prev_dim(2) + obj.padding(2) * 2 + obj.patch(1), iclust_sz);
                    X_(:, (obj.padding(1) + 1):(end - obj.patch(1) - obj.padding(1)), (obj.padding(2) + 1):(end - obj.patch(1) - obj.padding(2)), :) = X(:, :, :, idx_in);
                    stacked = zeros(size(X, 1) * prod(obj.dims(1:2)), iclust_sz, obj.patch(1) * obj.patch(2));
                    for x = 1:obj.dims(1)
                        for y = 1:obj.dims(2)
                            sx = (x - 1) * obj.stride(1) + 1;
                            ex = sx + obj.patch(1) - 1;
                            sy = (y - 1) * obj.stride(2) + 1;
                            ey = sy + obj.patch(2) - 1;
                            tmp = permute(X_(:, sx:ex, sy:ey, :), [1, 4, 2, 3]);
                            idx = ((y - 1) * obj.dims(1) + x - 1) * bs + 1;
                            stacked(idx : (idx + bs - 1), :, :) = tmp(:, :, :);
                        end
                    end
                    for r = 1 : obj.rank
                        comb = zeros(size(stacked, 1), size(stacked, 3));
                        % 1. Linearly combine input maps
                        for c = 1 : iclust_sz
                            comb = comb + squeeze(stacked(:, c, :)) * v.C(oc, c, r, ic);
                        end
                        % 2. Mult & sum with filter
                        comb = comb * v.XY(oc, :, r, ic)';
                        % 3. Mult by output filter coeff
                        out = zeros(size(comb, 1), oclust_sz);
                        for f = 1 : oclust_sz
                            out(:, f) = comb .* v.F(oc, f, r, ic);
                        end
                        res = res + out;
                    end
                    fprintf('oc = %d, ic = %d\n', oc, ic);
                end
                final_res(:, idx_out) = res;
            end
            results = reshape(final_res, [bs, obj.dims(1:2), obj.depth]);           
            results = bsxfun(@plus, results, reshape(v.B, [1, 1, 1, length(v.B)]));
            obj.cpu.vars.forward_act = results;              
            obj.cpu.vars.out = obj.F(results);
        end     
        
        function BP(obj)
            assert(0);
        end
        
        % XXX: allocate forward_act var and other missing.
        function InitWeights(obj)          
            global plan
            prev_dim = obj.prev_dim();
            dim = obj.dims();
            obj.AddParam('B', [obj.depth, 1], false);  
            obj.AddParam('F', [obj.oclust, dim(3) / obj.oclust, obj.rank, obj.iclust], false);          
            obj.AddParam('C', [obj.oclust, prev_dim(3) / obj.iclust, obj.rank, obj.iclust], false);   
            obj.AddParam('XY', [obj.oclust, prod(obj.patch), obj.rank, obj.iclust], false);     
            obj.AddParam('perm_in', [prev_dim(3)], false); 
            obj.AddParam('perm_out', [dim(3)], false); 
        end
    end
end

function json = FillDefault(json)
json.type = 'MonoConv';
end
