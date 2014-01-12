classdef MockApprox < Approximation
    properties        
        nr_execution
    end
    
    methods
        function obj = MockApprox(suffix, approx_vars, cuda_vars)
            obj@Approximation(suffix, approx_vars, cuda_vars);
            obj.name = 'mock_approx';
            obj.nr_execution = 0;
        end
        
        function ret = VerifyCombination(obj, approx_vars, cuda_vars)
            % Arbitrary: approx_vars.A must be divisible by cuda_vars.X
            if (mod(approx_vars.A, cuda_vars.B) == 0)
                ret = true;
                obj.nr_execution = obj.nr_execution + 1;
            else
                ret = false;                
            end
            printf(2, 'Calling VerifyCombination with %s, %s, ret = %d\n', struct2str(approx_vars), struct2str(cuda_vars), ret);            
        end        
        
        function [Wapprox, ret] = Approx(obj, params)   
            global plan
            Wapprox = 2 * plan.layer{2}.cpu.vars.W;
            ret = struct();            
            ret.layer = 'MockFC';
            ret.json = struct();
        end        
    end
end
