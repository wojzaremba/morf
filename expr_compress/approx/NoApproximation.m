classdef NoApproximation < Approximation
    properties        
    end
    
    methods
        function obj = NoApproximation(suffix, approx_vars, cuda_vars)
            obj@Approximation(suffix, approx_vars, cuda_vars);
            obj.name = 'no_approximation';
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            ret = true;
        end        
        
        function [Wapprox, ret] = Approx(obj, params)            
            global plan;
            ret.layer_nr = 2;
            ret.layer = 'Conv';
            ret.json = struct();
            ret.vars = struct(); 
            Wapprox = plan.layer{2}.cpu.vars.W;
            ret.on_gpu = Val(params, 'on_gpu', obj.on_gpu);
        end

    end
    
end

