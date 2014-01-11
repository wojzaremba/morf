classdef Approximation < handle
    properties 
        name
        iter_approx_vars
        iter_cuda_vars
        approx_vars
        cuda_vars        
    end
    
    methods
        function obj = Approximation(approx_vars, cuda_vars)
            obj.approx_vars = approx_vars;
            obj.cuda_vars = cuda_vars;            
            obj.iter_approx_vars = 1;
            obj.iter_cuda_vars = 1;
        end
        
        function [Wapprox, args] = Approx(obj, params)
            assert(0);
        end
        
        function [success, params] = GetApproxVars(obj)            
            if obj.iter_approx_vars <= length(obj.approx_vars) 
                success = 1;
                params = obj.approx_vars(obj.iter_approx_vars);
                printf(2, 'Getting %d ApproxVars, success = %d\n', obj.iter_approx_vars, success);
                obj.iter_approx_vars = obj.iter_approx_vars + 1;
            else
               success = 0;
               params = [];
               printf(2, 'Out of range ApproxVars, success = %d\n', success);
            end
            
        end
        
        function [success, cuda_vars] = GetCudaVars(obj, approx_vars)
            for i = obj.iter_cuda_vars : length(obj.cuda_vars)
                cuda_vars = obj.cuda_vars(i);
                if (~obj.VerifyCombination(approx_vars, cuda_vars))   
                    continue;
                end          
                success = 1;                                    
                printf(2, 'Getting %d GetCudaVars, success = %d\n\n', obj.iter_cuda_vars, success);
                obj.iter_cuda_vars = i + 1;
                return;
            end
            obj.iter_cuda_vars = length(obj.cuda_vars) + 1;
            success = 0;
            cuda_vars = [];            
        end
        
        function ret = CudaTemplate(obj)
            global root_path
            ret = sprintf('%s/expr_compress/cuda/src/%s_template.cuh', root_path, obj.name);
        end

        function ret = CudaTrue(obj)
            global root_path
            ret = sprintf('%s/expr_compress/cuda/src/%s_gen.cuh', root_path, obj.name);
        end        
        
        function VerifyCombination(obj, approx_vars, cuda_vars)
            assert(0);
        end
        
        function ResetApproxVarsIter(obj)
            obj.iter_approx_vars = 1;
        end
        function ResetCudaVarsIter(obj)
            obj.iter_cuda_vars = 1;
        end
        
        function [Wapprox, ret] = ApproxGeneric(obj, params)
            [Wapprox, ret] = obj.Approx(params);
            ret.cuda_template = obj.CudaTemplate();
            ret.cuda_true = obj.CudaTrue();
            ret.approx_params = params;            
        end
        
    end
end