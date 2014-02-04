classdef LRNormal < Layer
    properties
        k
        n
        alpha
        beta
    end
    
    methods
        function obj = LRNormal(json)
            obj@Layer(FillDefault(json));
            obj.k = single(json.k);
            obj.n = single(json.n);
            obj.alpha = single(json.alpha);
            obj.beta = single(json.beta);
            obj.Finalize();
            LRNormalCpp();
        end
        
        function FP_(obj)
            v = obj.gpu.vars;
            C_(ConvResponseNormCrossMap, v.X, v.denoms, v.out, obj.depth(), obj.n, obj.k, obj.alpha, obj.beta);
        end
        
        function FPmatlab(obj)
            X = obj.cpu.vars.X;
            normal = zeros(size(X), class(X));
            for i = 1:obj.depth
                normal(:, :, :, i) = obj.k + obj.alpha * sum(X(:, :, :, max(i - (obj.n - 1) / 2, 1):min(i + (obj.n - 1) / 2, obj.depth)) .^ 2, 4);
            end            
            obj.cpu.vars.normal = normal;
            obj.cpu.vars.out = X ./ (normal .^ obj.beta);
        end
        
        function FP(obj)
            v = obj.cpu.vars; 
            LRNormalCpp(v.X, v.out, obj.k, obj.n, obj.alpha, obj.beta);
        end        
        
        function BP(obj)
            v = obj.cpu.vars;
            data = obj.cpu.dvars.out;
            bs = size(data, 1);
            normal = obj.cpu.vars.normal;
            dX = zeros(size(v.X), class(v.X));            
            for b = 1:bs
                for i = 1:obj.depth
                    dX(b, :, :, i) = ((normal(b, :, :, i) .^ obj.beta) - 2 .* (v.X(b, :, :, i) .^ 2) .* obj.alpha .* obj.beta .* (normal(b, :, :, i) .^ (obj.beta - 1))) ./ (normal(b, :, :, i) .^ (2 * obj.beta));
                    dX(b, :, :, i) = dX(b, :, :, i) .* data(b, :, :, i);
                    for j = max(i - (obj.n - 1) / 2, 1):min(i + (obj.n - 1) / 2, obj.depth)
                        if (j == i)
                            continue;
                        end
                        dX(b, :, :, i) = dX(b, :, :, i) - ...
                            data(b, :, :, j) .* (2 .* v.X(b, :, :, j) .* v.X(b, :, :, i) .* obj.alpha .* obj.beta .* (normal(b, :, :, j) .^ (obj.beta - 1))) ./ (normal(b, :, :, j) .^ (2 * obj.beta));
                    end
                end
            end
            obj.cpu.dvars.X = dX;
        end
        
        function InitWeights(obj)
            global plan
            obj.AddParam('denoms', [plan.input.batch_size, obj.dims], false);  % XXX: this is incorrect
        end
    end
end

function json = FillDefault(json)
json.type = 'LRNormal';
json.one2one = true;
end
