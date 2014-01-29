classdef Slowness < Layer
    properties
        lambda
        slowness
        out
    end
    
    methods
        function obj = Slowness(json)
            obj@Layer(json);
            obj.lambda = Val(json, 'lambda', 1);
            obj.slowness = Val(json, 'slowness', 1);
            obj.Finalize();
        end
        
        function out = FP(layer)
            X = layer.params.X(:, :);  
            out = zeros(size(X, 1), 1);            
            if (layer.slowness == 3)
                range = size(X, 1) / 3;
                out(1:range) = layer.lambda * sum((X(1:range, :) + X((2 * range + 1):end, :) - 2 * X((range + 1):(2 * range), :)) .^ 2, 2);
            elseif (layer.slowness == 2)
                range = size(X, 1) / 2;
                out(1:range) = layer.lambda * sum((X(1:range, :) - X((range + 1):(2 * range), :)) .^ 2, 2);                
            end
            layer.params.out = out;
        end
        
        function dparams = BP(layer, node_data)
            X = layer.params.X(:, :);
            dX = zeros(size(X));
            if (layer.slowness == 3)
                range = size(X, 1) / 3;            
                dX(1:range, :) = 2 * (X(1:range, :) + X((2 * range + 1):end, :) - 2 * X((range + 1):(2 * range), :)) .* repmat(reshape(node_data(1:range), range, 1), [1, size(X(:, :), 2)]);
                dX((2 * range + 1):end, :) = dX(1:range, :);
                dX((range + 1):(2 * range), :) = -2 * dX(1:range, :);
            elseif (layer.slowness == 2)
                range = size(X, 1) / 2;            
                dX(1:range, :) = 2 * (X(1:range, :) - X((range + 1):(2 * range), :)) .* repmat(reshape(node_data(1:range), range, 1), [1, size(X(:, :), 2)]);
                dX((range + 1):(2 * range), :) = -dX(1:range, :);                
            end
            dX = layer.lambda * dX;
            dparams = struct('dX', dX);
        end
        
        function InitWeights(obj)
        end
    end
end
