classdef CudasLog < handle
    properties
        test_error
        cuda_results % Hashmap: maps cuda parameters to cuda results
    end
    
    % XXX : get a gpu name.
    
    methods
        function obj = CudasLog(info)            
            obj.test_error = info.test_error;        
            obj.cuda_results = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end
        
        function AddCudaExecutionResults(obj, cuda_params, cuda_results)
            cuda_str = struct2str(cuda_params);
            if (obj.cuda_results.isKey(cuda_str))
                fprintf('Uh oh: trying to overwrite cuda execution results.\n');
                assert(0);
            end
            obj.cuda_results(cuda_str) = cuda_results;
        end
        
        function Printf(obj)
            key_set = keys(obj.cuda_results);
            for i = 1 : length(key_set)
                key = key_set{i};
                fprintf('\t%s ', key);
                cuda_result = obj.cuda_results(key);
                fprintf('\t\tcuda_eq = %d\n\t\tcuda_test_error = %f\n\t\tcuda_speedup = %f \n', cuda_result.cuda_eq, cuda_result.cuda_test_error, cuda_result.cuda_speedup);
            end
        end
        
        function [best_cuda_vars, best_speedup] = BestSpeedup(obj)
            key_set = keys(obj.cuda_results);
            best_speedup = 0;
            best_cuda_vars = '';
            for i = 1 : length(key_set)
                key = key_set{i};
                speedup = obj.cuda_results(key).cuda_speedup;
                if (speedup > best_speedup)
                    best_speedup = speedup;
                    best_cuda_vars = key;
                end
            end
        end
        
    end
    
end