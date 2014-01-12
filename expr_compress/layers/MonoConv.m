classdef MonoConv < Layer
    properties
    end
    
    methods
        function obj = MonoConv(json)
            obj@Layer(FillDefault(json));
            obj.Finalize();
        end                      
       
        function FP_(obj)
            global plan
            v = obj.gpu.vars;            
            pdims = obj.prev_dim();            
            Capprox_gen(approx_pointer, v.Xmono, v.Wmono, v.out_mono, pdims(2), pdims(3), obj.patch(1), obj.stride(1), obj.padding(1), v.perm);
            Capprox_gen(Reshape, v.out, bs * obj.dims(1) * obj.dims(2), obj.depth());
            Capprox_gen(AddVector, v.out, v.B, v.out);
            Capprox_gen(Reshape, v.out, obj.dims(1) * obj.dims(2) * obj.depth(), bs);
            Capprox_gen(obj.Fun_, v.out, v.out);                  
        end
        
        function FP(obj)
            assert(0);
        end     
        
        function BP(obj)
            assert(0);
        end
        
        % XXX: allocate forward_act var and other missing.
        function InitWeights(obj)
            assert(0);
            global plan
            prev_dim = obj.prev_dim();
            obj.AddParam('B', [obj.depth, 1], true);  
            obj.AddParam('W', [obj.depth, obj.patch(1), obj.patch(2), prev_dim(3)], true);            
        end
    end
end

function json = FillDefault(json)
json.type = 'MonoConv';
end
