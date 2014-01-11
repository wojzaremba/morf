% XXX : missing serialization.

classdef ApproxLog < handle
    properties
        name % Name of approximation for which this log is defined.
        cudas_logs % Hashmap: maps approximation parameters to CudasLog object
    end
    
    methods
        function obj = ApproxLog(name)
            obj.name = name;
            obj.cudas_logs = containers.Map('KeyType', 'char', 'ValueType', 'any');
        end              
        
        function ret = IsProcessed(obj, approx_vars, cuda_vars)
            % Are approx_vars key in map? If not, retunr false
            approx_str = struct2str(approx_vars);
            if (~obj.cudas_logs.isKey(approx_str))
                ret = false;
                return;
            end
            % approx_vars are in map, so how about cuda_vars?
            cudas_log = obj.cudas_logs(approx_str);
            ret = cudas_log.IsProcessed(cuda_vars);
        end
        
        function SaveApproxInfo(obj, approx_vars, info)
            approx_str = struct2str(approx_vars);
            if (obj.cudas_logs.isKey(approx_str))
                fprintf('There is a bug !!!\n');
                assert(0);
            end
            obj.cudas_logs(approx_str) = CudasLog(info);
        end
        
        function ret = GetApproxInfo(obj, approx_vars)
            str = struct2str(approx_vars);
            if (~obj.cudas_logs.isKey(str))
                fprintf('There should be value stored\n');
                assert(0);
            end
            ret = obj.cudas_logs(str);
        end
        
        function AddCudaExecutionResults(obj, approx_vars, cuda_vars, cuda_results)
           approx_str = struct2str(approx_vars);
           if (~obj.cudas_logs.isKey(approx_str))
                fprintf('Uh oh: trying to add cuda results before saving approx info!\n');
                assert(0);
           end
           cuda_log = obj.cudas_logs(approx_str);
           cuda_log.AddCudaExecutionResults(cuda_vars, cuda_results);
        end
                
        function Printf(obj)
            fprintf('Approximation = %s\n', obj.name);
            cudas_logs = obj.cudas_logs;
            key_set = keys(cudas_logs);
            for i = 1 : length(key_set)
                key = key_set{i};
                fprintf('Approx vars: \t%s \n', key);
                cudas_log = cudas_logs(key);
                cudas_log.Printf();
            end
        end
    end
    
end