classdef Sum < Layer
    properties
        lambda
        hidden
    end
    
    methods
        function obj = Sum(json)
            obj@Layer(json);
            obj.Finalize();
        end
        
        function FP(layer)
            X = layer.params.X(:, :);
            layer.params.out = sum(X, 2);
        end
        
        function dparams = BP(layer, node_data)
            X = layer.params.X(:, :);
            dparams = struct('dX', repmat(node_data, [1, size(X, 2)]));
        end
        
        function InitWeights(obj)    
        end
    end
end
