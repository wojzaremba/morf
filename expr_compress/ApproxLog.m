classdef ApproxLog < handle
    properties
        name % Name of approximation for which this log is defined.
        cudas_logs % Hashmap: maps approximation parameters to CudasLog object
        scheduler_params
    end
    
    methods
        function obj = ApproxLog(name, scheduler_params)
            obj.name = name;
            try
                logs = load(obj.LogName());
                obj.cudas_logs = logs.cudas_logs;
                printf(2, 'Opened log file %s\n', obj.LogName());
                
            catch
                obj.cudas_logs = containers.Map('KeyType', 'char', 'ValueType', 'any');        
                printf(2, 'Failed to open log file %s\n', obj.LogName());
            end       
            obj.scheduler_params = scheduler_params;
        end              
        
        function str = LogName(obj)
            global root_path
            str = sprintf('%sexpr_compress/logs/%s', root_path, obj.name);
        end
        
        function ClearLog(obj)
            delete(obj.LogName());
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
        
        function Save(obj)
            global root_path
            cudas_logs = obj.cudas_logs;
            dir = sprintf('%sexpr_compress/logs', root_path);
            if (~exist(dir, 'dir'))
                mkdir(dir);
            end
            save(obj.LogName(), 'cudas_logs');            
        end
        
        function SaveApproxInfo(obj, approx_vars, info)
            approx_str = struct2str(approx_vars);
            if (obj.cudas_logs.isKey(approx_str))
                fprintf('Resaving CudasLog results\n');
                assert(obj.cudas_logs(approx_str).test_error == info.test_error);                
            end
            obj.cudas_logs(approx_str) = CudasLog(info);
            obj.Save();
        end
        
        function [ret, present] = GetApproxInfo(obj, approx_vars)
            str = struct2str(approx_vars);           
            if (~obj.cudas_logs.isKey(str))
                fprintf('Reading from the empty approx log\n');
                present = false;
                ret = [];                
            else
                present = true;
                ret = obj.cudas_logs(str);
            end            
        end
        
        function AddCudaExecutionResults(obj, approx_vars, cuda_vars, cuda_results)
           approx_str = struct2str(approx_vars);
           if (~obj.cudas_logs.isKey(approx_str))
                fprintf('Uh oh: trying to add cuda results before saving approx info!\n');
                assert(0);
           end
           cuda_log = obj.cudas_logs(approx_str);
           cuda_results.no_compilation = obj.scheduler_params.no_compilation;
           cuda_log.AddCudaExecutionResults(cuda_vars, cuda_results);
           Save(obj);
        end
        
        function VerifyCompiled(obj)
            cudas_logs = obj.cudas_logs;
            key_set = keys(cudas_logs);
            for i = 1 : length(key_set)
                key = key_set{i};
                cudas_log = cudas_logs(key);
                assert(cudas_log.no_compilation == 0);
            end            
        end
                
        function Printf(obj)
            fprintf('Approximation = %s\n', obj.name);
            cudas_logs = obj.cudas_logs;
            key_set = keys(cudas_logs);
            for i = 1 : length(key_set)
                key = key_set{i};
                fprintf('\tApprox vars: %s \n', key);
                cudas_log = cudas_logs(key);
                cudas_log.Printf();
            end
        end
        
        function [best_cuda_vars, best_speedups] = BestSpeedup(obj)
            cudas_logs = obj.cudas_logs;
            key_set = keys(cudas_logs);
            best_cuda_vars = [];
            best_speedups = [];
            for i = 1 : length(key_set)
                key = key_set{i};
                cudas_log = cudas_logs(key);
                [cuda_vars, speedup] = cudas_log.BestSpeedup();
                best_cuda_vars = [best_cuda_vars, cuda_vars];
                best_speedups = [best_speedups, speedup];
                fprintf('\tApprox vars: %s \n', key);
                fprintf('\t\tBest Speedup: %f with cuda_vars = %s\n\n', speedup, cuda_vars);
            end
        end
    end
    
end