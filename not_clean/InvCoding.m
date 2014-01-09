classdef InvCoding < Layer
    properties
        alpha
        lambda        
    end
    
    methods
        function obj = InvCoding(json)
            obj@Layer(json);         
            obj.lambda = Val(json, 'lambda', 0.1);
            obj.alpha = Val(json, 'alpha', 1);
            obj.Finalize();
        end
        
        function FP(layer)
            X = layer.params.X(:, :);
            range = size(X, 1) / 2;
            X1 = X(1:range, :);
            X2 = X((range + 1):end, :);
            S = layer.params.S(:, :);
            dist = S' * S - eye(layer.depth());
            out = zeros(size(X, 1), layer.depth());
            out(1:range, :) = ((X1 - X2) * S) .^ 2 - layer.alpha * (X2 * S) .^ 2 + layer.lambda * sum(dist(:) .^ 2);
            layer.params.out = out;
        end
        
        function dparams = BP(layer, ~)
            X = layer.params.X(:, :);            
            range = size(X, 1) / 2;
            X1 = X(1:range, :);
            X2 = X((range + 1):end, :);            
            S = layer.params.S(:, :);
            dist = S' * S - eye(layer.dims(1));               
            
            dS = 2 * squeeze(sum(repmat(reshape((X1 - X2) * S, [range, 1, layer.dims(1)]), [1, size(X, 2), 1]) .* repmat(X1 - X2, [1, 1, layer.depth()]), 1)); 
            dS = dS - layer.alpha * 2 * squeeze(sum(repmat(reshape(X2 * S, [range, 1, layer.depth()]), [1, size(X, 2), 1]) .* repmat(X2, [1, 1, layer.depth()]), 1)); 
            dS = dS + layer.lambda * 16 * range * squeeze(sum(repmat(reshape(dist, [1, layer.depth(), layer.depth()]), [size(S, 1), 1, 1]) .* repmat(S, [1, 1, size(S, 2)]), 2));
            dparams = struct('dS', reshape(dS, size(S)));
        end
        
        function InitWeights(obj)
            S = zeros(prod(obj.prev_dim()), obj.depth());   
            for i = 1:size(S, 1)
                S(i, mod(i, size(S, 2)) + 1) = 1;
            end
            S = S + randn(size(S)) / 10;
            obj.params.S = S;
        end
    end
end
