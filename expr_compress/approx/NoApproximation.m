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
            Wapprox = plan.layer{2}.cpu.vars.W;          
            ret.Wapprox = Wapprox;
        end
        
        function [test_error, time] = RunModifConv(obj, args)        
            global plan          
            ForwardPass(plan.input);
            test_error = plan.classifier.GetScore();            
            time = plan.time.fp(2);
        end
    end
    
end