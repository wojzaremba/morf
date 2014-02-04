classdef MaxPooling < Layer
    properties
    end
    
    methods
        function obj = MaxPooling(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();            
            MaxPoolCpp();
        end
        
        function FP_(obj)
            v = obj.gpu.vars;
            C_(MaxPool, v.X, v.out, obj.depth(), obj.patch(1), obj.stride(1), obj.dims(1));            
        end
        
        function FPmatlab(obj)
            X = obj.cpu.vars.X;
            dims = obj.dims;
            dims_prev = obj.prev_dim();
            bs = size(X, 1);
            out = zeros([bs, dims(3), dims(1), dims(2)], class(X));
            idx = zeros([bs, dims(3), dims(1), dims(2)], class(X));
            X = permute(X, [1, 4, 2, 3]);
            X = reshape(X, [bs, dims_prev(3), dims_prev(1),  dims_prev(2)]);
            for b = 1:dims(1)
                for c = 1:dims(2)
                    sx = (b - 1) * obj.stride(1) + 1;
                    sy = (c - 1) * obj.stride(2) + 1;
                    tmp = X(:, :, sx:min(sx + obj.patch(1) - 1, size(X, 3)), sy:min(sy + obj.patch(2) - 1, size(X, 4)));
                    [out(:, :, b, c), idx(:, :, b, c)] = max(tmp(:, :, :), [], 3);
                end
            end         
            out = permute(out, [1, 3, 4, 2]);
            obj.cpu.vars.idx = idx;
            obj.cpu.vars.out = out;
        end
        
        function FP(obj)
            v = obj.cpu.vars; 
            MaxPoolCpp(v.X, v.out, obj.patch(1), obj.stride(1));
        end        
        
        function BP(obj)
            global plan;
            data = obj.cpu.dvars.out;
            X = obj.cpu.vars.X;
            dX = zeros([size(X, 1), size(X, 4), size(X, 2), size(X, 3)], class(X));
            idx = obj.cpu.vars.idx;
            idx1 = mod(idx - 1, obj.patch(2));
            idx2 = floor((idx - 1) / obj.patch(2));
            dims = obj.dims;
            bs = plan.input.batch_size;
            data = reshape(data, [bs, dims]);
            for b = 1:dims(1)
                for c = 1:dims(2)
                    sx = (b - 1) * obj.stride(1) + 1;
                    sy = (c - 1) * obj.stride(2) + 1;
                    tx = idx1(:, :, b, c) + sx;
                    ty = idx2(:, :, b, c) + sy;                    
                    idx_ = repmat(1:obj.depth(), bs, 1) + (tx - 1) * obj.depth() + (ty - 1) * obj.depth() * size(dX, 3);                    
                    idx_dX = (idx_ - 1) * bs + repmat((1:bs)', [1, size(idx_, 2)]);
                    dX(idx_dX) = dX(idx_dX) + reshape(data(:, b, c, :), bs, dims(3));
                end
            end
            obj.cpu.dvars.X = permute(dX, [1, 3, 4, 2]);
        end
        
        function InitWeights(obj)
        end        
    end
end

function json = FillDefault(json)
json.type = 'MaxPooling';
end