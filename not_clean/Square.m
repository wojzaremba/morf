classdef Square < Layer
    properties
        out
    end
    
    methods
        function obj = Square(json)
            obj@Layer(json);
            obj.Finalize();
            global plan
            plan.classifier = obj;            
        end
        
        function out = FP(layer)
            X = layer.params.X(:, :);  
            out = X;
            layer.out = out;
        end
        
        function dparams = BP(layer, data)
            dX = 2 * (layer.out - data);
            dparams = struct('dX', dX);            
        end
        
        function incorrect = GetScore(layer, Y)
          incorrect = mean((layer.out(:) - Y(:)) .^ 2);
        end              
        
        function InitWeights(obj)
            obj.params = struct();
            obj.InitGrads();
        end
    end
end
