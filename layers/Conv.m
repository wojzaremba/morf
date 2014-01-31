classdef Conv < Layer
    properties
    end
    
    methods
        function obj = Conv(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end                      
       
        function FP_(obj)
            global plan
            bs = plan.input.batch_size;
            v = obj.gpu.vars;
            pdims = obj.prev_dim();
            C_(ConvAct, v.X, v.W, v.out, pdims(2), pdims(3), obj.patch(1), obj.stride(1), obj.padding(1));
            C_(Reshape, v.out, bs * obj.dims(1) * obj.dims(2), obj.depth());
            C_(AddVector, v.out, v.B, v.out);
            C_(Reshape, v.out, obj.dims(1) * obj.dims(2) * obj.depth(), bs);
            C_(obj.Fun_, v.out, v.out);            
        end
        
        function FP_old(obj)
            global plan
            prev_dim = obj.prev_dim();
            v = obj.cpu.vars;
            X = v.X;            
            bs = size(X, 1);
            X_ = zeros(size(X, 1), prev_dim(1) + obj.padding(1) * 2 + obj.patch(1), prev_dim(2) + obj.padding(2) * 2 + obj.patch(1), prev_dim(3), class(X));
            X_(:, (obj.padding(1) + 1):(obj.padding(1) + size(X, 2)), (obj.padding(2) + 1):(obj.padding(2) + size(X, 3)), :) = X;
            stacked = zeros(size(X, 1) * prod(obj.dims(1:2)), obj.patch(1) * obj.patch(2) * prev_dim(3), class(X));
            for x = 1:obj.dims(1)
                for y = 1:obj.dims(2)
                    sx = (x - 1) * obj.stride(1) + 1;
                    ex = sx + obj.patch(1) - 1;
                    sy = (y - 1) * obj.stride(2) + 1;
                    ey = sy + obj.patch(2) - 1;
                    tmp = X_(:, sx:ex, sy:ey, :);
                    idx = ((y - 1) * obj.dims(1) + x - 1) * bs + 1;
                    stacked(idx : (idx + bs - 1), :) = tmp(:, :);
                end
            end
            results = stacked * v.W(:, :)';
            results = reshape(results, [bs, obj.dims(1:2), obj.depth]);           
            results = bsxfun(@plus, results, reshape(v.B, [1, 1, 1, length(v.B)]));
            if (plan.only_fp == 0)            
                obj.cpu.vars.forward_act = results;              
                obj.cpu.vars.X_ = X_;
                obj.cpu.vars.stacked = stacked;
            end
            obj.cpu.vars.out = obj.F(results);
        end         
          
        function FP(obj)
            v = obj.cpu.vars; 
            ConvCpp(v.X, v.W, v.B, v.out, obj.stride(1), obj.padding(1));
        end
        
        function BP(obj)
            global plan
            v = obj.cpu.vars;
            data = obj.cpu.dvars.out;
            stacked = v.stacked;
            W = v.W;
            X = v.X;
            act = v.forward_act;
            act = obj.dF(act);
            nact = act .* reshape(squeeze(data), size(act));
            pact = permute(nact, [4, 1, 2, 3]);
            bs = size(data, 1);
            if (obj.bp_dx_on_first == 1) || (obj.layer_nr >= 3)
                dX_ = zeros(size(X, 1), size(X, 2) + obj.padding(1) * 2 + obj.patch(1), size(X, 3) + obj.padding(2) * 2 + obj.patch(2), size(X, 4));                
                for x = 1:obj.dims(1)
                    sx = (x - 1) * obj.stride(1) + 1;
                    ex = sx + obj.patch(1) - 1;                    
                    for y = 1:obj.dims(2)
                        sy = (y - 1) * obj.stride(2) + 1;
                        ey = sy + obj.patch(2) - 1;
                        idx = ((y - 1) * obj.dims(1) + x - 1) * bs + 1;
                        dX_(:, sx:ex, sy:ey, :) = dX_(:, sx:ex, sy:ey, :) + reshape(pact(:, idx : (idx + bs - 1))' * W(:, :), size(dX_(:, sx:ex, sy:ey, :)));
                    end
                end
                obj.cpu.dvars.X = dX_(:, (obj.padding(1) + 1):(obj.padding(1) + size(X, 2)), (obj.padding(2) + 1):(obj.padding(2) + size(X, 3)), :);                
            end
            obj.cpu.dvars.W = reshape(pact(:, :) * stacked, size(W));
            obj.cpu.dvars.B = reshape(sum(pact(:, :), 2), size(v.B));
        end
        
        function InitWeights(obj)
            global plan
            prev_dim = obj.prev_dim();
            obj.AddParam('B', [obj.depth, 1], true);  
            obj.AddParam('W', [obj.depth, obj.patch(1), obj.patch(2), prev_dim(3)], true);            
        end
    end
end

function json = FillDefault(json)
json.type = 'Conv';
end
