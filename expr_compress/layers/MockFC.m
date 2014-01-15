classdef MockFC < LayerApprox
    properties
    end
    
    methods
        function obj = MockFC(json)
            obj@LayerApprox(FillDefault(json));
            obj.Finalize();
        end                      
       
        function FP_(obj)
            global executed_mock_fc
            executed_mock_fc = executed_mock_fc + 1;
            v = obj.gpu.vars;
            Capprox_gen(Mult, v.X, v.Wmock, v.out);
            Capprox_gen(Scale, v.out, 10000, v.out);
            Capprox_gen(AddVector, v.out, v.Bmock, v.out);
            Capprox_gen(obj.Fun_, v.out, v.out);            
        end
        
        function FP(obj)
            assert(0);
        end     
        
        function BP(obj)
            assert(0);
        end
        
        function InitWeights(obj)
            obj.AddParam('Bmock', [1, prod(obj.dims)], true);
            obj.AddParam('Wmock', [prod(obj.prev_dim()), prod(obj.dims)], true);                        
        end
    end
end

function json = FillDefault(json)
json.type = 'MockFC';
end
