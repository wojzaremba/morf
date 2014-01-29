classdef CudasLog < handle
    properties
        test_error
        orig_time
        cuda_results % Hashmap: maps cuda parameters to cuda results
        no_compilation
    end
    
    % XXX : get a gpu name.
    
    methods
        function obj = CudasLog(info)            
            obj.test_error = info.test_error;        
            obj.orig_time = info.orig_time;   
%             XXX : This is incorrect, we should take care of
%             no_compilation.
%             obj.no_compilation = info.no_compilation;
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
        
        function ret = IsProcessed(obj, cuda_vars)
            % Are cuda_vars in map?
            cuda_str = struct2str(cuda_vars);
            ret = obj.cuda_results.isKey(cuda_str);
        end
        
        function Printf(obj)
            fprintf('\ttest_error = %d \n', obj.test_error);
            fprintf('\torig_time = %f \n', obj.orig_time);
            key_set = keys(obj.cuda_results);
            for i = 1 : length(key_set)
                key = key_set{i};
                fprintf('\tCuda vars: %s \n', key);
                cuda_result = obj.cuda_results(key);
                fprintf('\t\ttime = %f\n', cuda_result.approx_time);
            end
        end
        
        function [best_cuda_vars, best_speedup] = BestSpeedup(obj)
            key_set = keys(obj.cuda_results);
            best_speedup = 0;
            best_cuda_vars = '';
            for i = 1 : length(key_set)
                key = key_set{i};
                speedup = obj.cuda_results(key).approx_time;
                if (speedup > best_speedup)
                    best_speedup = speedup;
                    best_cuda_vars = key;
                end
            end
            best_speedup = obj.orig_time / speedup;
        end
        
    end
    
end