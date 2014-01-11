classdef MockApprox < Approximation
    properties        
        nr_execution
    end
    
    methods
        function obj = MockApprox(approx_vars, cuda_vars)
            obj@Approximation(approx_vars, cuda_vars);
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
            Wapprox = params.A ^ 2 * ones(5, 1);
            ret = struct();
        end
        
        function ret = RunCuda(obj, args)
            ret.cuda_eq = 1;
            ret.cuda_test_error = 0;
            ret.cuda_speedup = 2;     
        end
    end
end
